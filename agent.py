"""Double DQN, кольцевой replay buffer и возобновляемые checkpoints."""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from config import AgentConfig
from environment import State
from model import QNetwork


class ReplayBuffer:
    """Массивы выделяются один раз; добавление O(1), выборка без повторений."""

    def __init__(self, capacity: int, state_dim: int, seed: int = 42) -> None:
        if min(capacity, state_dim) <= 0:
            raise ValueError("capacity и state_dim должны быть положительными")
        self.capacity = capacity
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminals = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, state: State, action: int, reward: float,
            next_state: State, terminated: bool) -> None:
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        # Time-limit truncation не является терминальным состоянием MDP:
        # для него сохраняем bootstrap от последнего состояния до reset.
        self.terminals[i] = terminated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        if not 0 < batch_size <= self.size:
            raise ValueError("Недостаточно переходов для выборки")
        indices = self.rng.choice(self.size, batch_size, replace=False)
        return tuple(torch.as_tensor(array[indices], device=device) for array in
                     (self.states, self.actions, self.rewards, self.next_states, self.terminals))

    def state_dict(self) -> dict[str, Any]:
        return {"capacity": self.capacity, "position": self.position, "size": self.size,
                "rng": json.dumps(self.rng.bit_generator.state),
                **{name: torch.from_numpy(getattr(self, name)[:self.size].copy())
                   for name in ("states", "actions", "rewards", "next_states", "terminals")}}

    def load_state_dict(self, data: dict[str, Any]) -> None:
        if data["capacity"] != self.capacity:
            raise ValueError("Емкость replay buffer отличается от checkpoint")
        self.position, self.size = int(data["position"]), int(data["size"])
        if not 0 <= self.size <= self.capacity or not 0 <= self.position < self.capacity:
            raise ValueError("Поврежденные индексы replay buffer")
        for name in ("states", "actions", "rewards", "next_states", "terminals"):
            getattr(self, name)[:self.size] = data[name].cpu().numpy()
        self.rng.bit_generator.state = json.loads(data["rng"])


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 config: AgentConfig | None = None, seed: int = 42) -> None:
        self.config = config or AgentConfig()
        c = self.config
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                                   if c.device == "auto" else c.device)
        self.state_dim, self.action_dim = state_dim, action_dim
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(state_dim, action_dim, c.hidden_sizes).to(self.device)
        self.target = QNetwork(state_dim, action_dim, c.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=c.learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(c.replay_capacity, state_dim, seed + 1)
        self.environment_steps = 0
        self.optimization_steps = 0

    @property
    def epsilon(self) -> float:
        c = self.config
        return c.epsilon_end + (c.epsilon_start - c.epsilon_end) * math.exp(
            -self.environment_steps / c.epsilon_decay)

    def act(self, state: State, *, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_dim))
        with torch.inference_mode():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def observe(self, state: State, action: int, reward: float,
                next_state: State, terminated: bool) -> None:
        self.replay.add(state, action, reward, next_state, terminated)
        self.environment_steps += 1

    def optimize(self) -> float | None:
        c = self.config
        if (len(self.replay) < max(c.learning_starts, c.batch_size)
                or self.environment_steps % c.train_frequency):
            return None
        states, actions, rewards, next_states, terminals = self.replay.sample(c.batch_size, self.device)
        predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if c.double_dqn:
                # Выбор действия online-сетью, оценка target-сетью снижает
                # систематическое завышение max Q в стандартном DQN.
                next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                next_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_values = self.target(next_states).max(dim=1).values
            expected = rewards + c.gamma * (~terminals).float() * next_values
        loss = self.loss_function(predicted, expected)
        if not torch.isfinite(loss):
            raise FloatingPointError("DQN loss стал NaN/Inf")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), c.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimization_steps += 1
        if self.optimization_steps % c.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: Path, *, metadata: dict[str, Any] | None = None,
             include_replay: bool = True) -> None:
        """Атомарная замена: прерванная запись не разрушает предыдущую копию."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": 1, "state_dim": self.state_dim, "action_dim": self.action_dim,
            "config": asdict(self.config), "online": self.online.state_dict(),
            "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
            "environment_steps": self.environment_steps, "optimization_steps": self.optimization_steps,
            "rng": json.dumps(self.rng.bit_generator.state), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "metadata": metadata or {},
        }
        if include_replay:
            payload["replay"] = self.replay.state_dict()
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, path: Path, *, resume: bool = True) -> dict[str, Any]:
        # weights_only запрещает произвольное исполнение pickle-классов.
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (data["version"], data["state_dim"], data["action_dim"]) != (1, self.state_dim, self.action_dim):
            raise ValueError("Несовместимый формат checkpoint или размеры модели")
        if resume:
            saved, current = dict(data["config"]), asdict(self.config)
            saved.pop("device", None)
            current.pop("device", None)
            if saved != current or "replay" not in data:
                raise ValueError("Для resume нужны одинаковые параметры и полный replay buffer")
        self.online.load_state_dict(data["online"])
        if resume:
            self.target.load_state_dict(data["target"])
            self.optimizer.load_state_dict(data["optimizer"])
            self.environment_steps = data["environment_steps"]
            self.optimization_steps = data["optimization_steps"]
            self.rng.bit_generator.state = json.loads(data["rng"])
            torch.set_rng_state(data["torch_rng"])
            if torch.cuda.is_available() and len(data["cuda_rng"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(data["cuda_rng"])
            self.replay.load_state_dict(data["replay"])
        return dict(data["metadata"])
