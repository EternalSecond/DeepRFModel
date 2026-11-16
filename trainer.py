"""CLI обучения, оценки и визуализации: python trainer.py --help."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent import DQNAgent
from config import AgentConfig, Config, EnvironmentConfig, TrainingConfig
from environment import ObstacleEnv

LOG = logging.getLogger(__name__)


def plot_metrics(history: list[dict[str, Any]], path: Path) -> None:
    """Agg работает на сервере без display; фигура всегда освобождается."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    episodes = [row["episode"] for row in history]
    rewards = np.asarray([row["reward"] for row in history])
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    try:
        axes[0].plot(episodes, rewards, alpha=0.35, label="Episode reward")
        window = min(25, len(history))
        mean = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(episodes[window - 1:], mean, label=f"Mean ({window})")
        evaluated = [row for row in history if row["eval_reward"] is not None]
        if evaluated:
            axes[0].plot([r["episode"] for r in evaluated], [r["eval_reward"] for r in evaluated], "o-", label="Greedy evaluation")
        axes[0].legend()
        axes[0].set_ylabel("Reward")
        axes[1].plot(episodes, [r["loss"] if r["loss"] is not None else np.nan for r in history])
        axes[1].set_ylabel("Huber loss")
        axes[2].plot(episodes, [r["progress"] for r in history], label="Progress")
        axes[2].plot(episodes, [r["epsilon"] for r in history], label="Epsilon")
        axes[2].legend()
        axes[2].set_xlabel("Episode")
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(path, dpi=140)
    finally:
        plt.close(figure)


class Trainer:
    def __init__(self, config: Config, output: Path, *, render: bool = False) -> None:
        self.config, self.output, self.render_enabled = config, output, render
        output.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(config.training.torch_threads)
        self.env = ObstacleEnv(config.environment, config.training.seed)
        self.agent = DQNAgent(config.environment.state_dim, config.environment.action_dim,
                              config.agent, config.training.seed)
        self.history: list[dict[str, Any]] = []
        self.best_reward = -float("inf")
        self.completed_episodes = 0

    def _metadata(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "history": self.history,
                "best_reward": self.best_reward, "completed_episodes": self.completed_episodes}

    def resume(self, checkpoint: Path) -> None:
        metadata = self.agent.load(checkpoint)
        if metadata["config"]["environment"] != asdict(self.config.environment):
            raise ValueError("Физика среды отличается от checkpoint")
        if metadata["config"]["training"]["seed"] != self.config.training.seed:
            raise ValueError("Для resume сохраните исходный seed")
        self.history = metadata["history"]
        self.best_reward = metadata["best_reward"]
        self.completed_episodes = metadata["completed_episodes"]

    def evaluate(self, episodes: int, *, render: bool = False) -> tuple[float, float]:
        env = ObstacleEnv(self.config.environment)
        rewards: list[float] = []
        successes = 0
        try:
            for episode in range(episodes):
                # Отдельный RNG, фиксированный набор трасс; epsilon и replay
                # при оценке не меняются. Для финального теста используйте иной seed.
                state, _ = env.reset(seed=self.config.training.seed + 1_000_000 + episode)
                total = 0.0
                while True:
                    state, reward, terminated, truncated, info = env.step(self.agent.act(state, explore=False))
                    total += reward
                    if render and not env.render(episode=episode + 1, reward=total, fps=round(1 / env.config.dt)):
                        raise KeyboardInterrupt
                    if terminated or truncated:
                        successes += int(bool(info["success"]))
                        break
                rewards.append(total)
        finally:
            env.close()
        return float(np.mean(rewards)), successes / episodes

    def _write_metrics(self) -> None:
        if self.history:
            with (self.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.history[0]))
                writer.writeheader()
                writer.writerows(self.history)
            plot_metrics(self.history, self.output / "training_progress.png")

    def train(self) -> None:
        t = self.config.training
        (self.output / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")
        LOG.info("Device: %s; output: %s", self.agent.device, self.output.resolve())
        try:
            for episode in range(self.completed_episodes + 1, t.episodes + 1):
                state, _ = self.env.reset(seed=t.seed + episode)
                total = 0.0
                losses: list[float] = []
                while True:
                    action = self.agent.act(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    self.agent.observe(state, action, reward, next_state, terminated)
                    loss = self.agent.optimize()
                    if loss is not None:
                        losses.append(loss)
                    total += reward
                    state = next_state
                    if self.render_enabled and not self.env.render(episode=episode, epsilon=self.agent.epsilon, reward=total):
                        raise KeyboardInterrupt
                    if terminated or truncated:
                        break
                evaluation_reward: float | None = None
                success_rate: float | None = None
                if episode % t.evaluation_interval == 0 or episode == t.episodes:
                    evaluation_reward, success_rate = self.evaluate(t.evaluation_episodes)
                self.completed_episodes = episode
                self.history.append({"episode": episode, "reward": total,
                    "loss": float(np.mean(losses)) if losses else None,
                    "epsilon": self.agent.epsilon, "steps": self.env.steps,
                    "progress": info["progress"], "reason": info["reason"],
                    "eval_reward": evaluation_reward, "eval_success_rate": success_rate})
                if evaluation_reward is not None and evaluation_reward > self.best_reward:
                    self.best_reward = evaluation_reward
                    self.agent.save(self.output / "best_model.pth", metadata=self._metadata(), include_replay=False)
                if episode % t.checkpoint_interval == 0:
                    self.agent.save(self.output / "last_checkpoint.pth", metadata=self._metadata())
                if episode % t.plot_interval == 0:
                    self._write_metrics()
                LOG.info("Episode %d | reward %.2f | progress %.1f%% | eps %.3f | %s",
                         episode, total, float(info["progress"]) * 100, self.agent.epsilon, info["reason"])
        except KeyboardInterrupt:
            # Частичный эпизод отсутствует в метриках; при resume его трасса
            # запускается заново, накопленные переходы остаются в replay.
            LOG.info("Остановка: сохраняю текущее состояние обучения")
        finally:
            self.env.close()
            self.agent.save(self.output / "last_checkpoint.pth", metadata=self._metadata())
            self._write_metrics()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=None, help="Общее число эпизодов (включая уже пройденные)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0")
    parser.add_argument("--output", type=Path, default=Path("runs/dqn"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--config", type=Path, help="JSON в формате config.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path)
    mode.add_argument("--evaluate", type=Path)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    raw: dict[str, Any] = {}
    if args.config:
        raw = json.loads(args.config.read_text(encoding="utf-8"))
    elif args.resume or args.evaluate:
        data = torch.load(args.resume or args.evaluate, map_location="cpu", weights_only=True)
        raw = data["metadata"]["config"]
        del data
    agent_data = dict(raw.get("agent", {}))
    if "hidden_sizes" in agent_data:
        agent_data["hidden_sizes"] = tuple(agent_data["hidden_sizes"])
    config = Config(EnvironmentConfig(**raw.get("environment", {})),
                    AgentConfig(**agent_data), TrainingConfig(**raw.get("training", {})))
    training = replace(config.training,
        episodes=args.episodes if args.episodes is not None else config.training.episodes,
        seed=args.seed if args.seed is not None else config.training.seed)
    config = replace(config, training=training,
                     agent=replace(config.agent, device=args.device or config.agent.device))
    trainer = Trainer(config, args.output, render=args.render)
    if args.evaluate:
        trainer.agent.load(args.evaluate, resume=False)
        try:
            reward, success = trainer.evaluate(args.episodes or training.evaluation_episodes, render=args.render)
            LOG.info("Evaluation | mean reward %.2f | success %.1f%%", reward, success * 100)
        except KeyboardInterrupt:
            LOG.info("Оценка остановлена")
        finally:
            trainer.env.close()
    else:
        if args.resume:
            trainer.resume(args.resume)
        trainer.train()


if __name__ == "__main__":
    main()
