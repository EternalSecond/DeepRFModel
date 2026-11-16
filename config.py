"""Единая, сериализуемая конфигурация симуляции и обучения (Python 3.11+)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvironmentConfig:
    width: int = 1000
    height: int = 600
    world_width: float = 3600.0
    wall: float = 24.0
    agent_radius: float = 12.0
    dt: float = 1.0 / 30.0
    acceleration: float = 850.0
    max_speed: float = 220.0
    drag: float = 3.5
    obstacle_count: int = 9
    obstacle_width: float = 36.0
    gap_height: float = 200.0
    ray_count: int = 16
    ray_length: float = 400.0
    max_steps: int = 2400
    stagnation_steps: int = 180
    progress_reward: float = 0.08
    survival_reward: float = 0.002
    idle_penalty: float = 0.03
    collision_penalty: float = 30.0
    stagnation_penalty: float = 15.0
    goal_reward: float = 100.0
    fps: int = 60

    @property
    def state_dim(self) -> int:
        # Лучи, x/y, vx/vy, расстояние до рекорда x, таймеры, расстояние до цели.
        return self.ray_count + 8

    @property
    def action_dim(self) -> int:
        return 5  # coast, right, up, down, left

    def __post_init__(self) -> None:
        positive = (self.width, self.height, self.world_width, self.agent_radius,
                    self.dt, self.acceleration, self.max_speed, self.ray_length,
                    self.max_steps, self.stagnation_steps, self.fps, self.obstacle_width)
        if min(positive) <= 0 or self.wall <= 0 or self.drag < 0:
            raise ValueError("Размеры, времена и скорости должны быть положительными")
        if self.ray_count < 4 or self.obstacle_count < 1:
            raise ValueError("Нужно минимум 4 луча и 1 препятствие")
        if not 4 * self.agent_radius < self.gap_height < self.height - 2 * self.wall:
            raise ValueError("Некорректная высота прохода")
        if self.world_width < self.width or self.world_width <= 800:
            raise ValueError("Длина мира должна быть >= ширины окна и > 800")
        if (self.world_width - 800) / max(1, self.obstacle_count - 1) <= self.obstacle_width + 4 * self.agent_radius:
            raise ValueError("Препятствия расположены слишком тесно")


@dataclass(frozen=True)
class AgentConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_starts: int = 2000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 80_000.0
    hidden_sizes: tuple[int, ...] = (256, 256)
    target_update_interval: int = 1000  # В шагах оптимизатора.
    train_frequency: int = 4
    gradient_clip: float = 10.0
    double_dqn: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("gamma и epsilon должны лежать в [0, 1]")
        if min(self.batch_size, self.replay_capacity, self.target_update_interval,
               self.train_frequency, self.epsilon_decay, self.learning_rate, self.gradient_clip) <= 0:
            raise ValueError("Параметры обучения должны быть положительными")
        if self.replay_capacity < max(self.batch_size, self.learning_starts):
            raise ValueError("Буфер меньше batch_size или learning_starts")
        if self.learning_starts < 0 or not self.hidden_sizes or min(self.hidden_sizes) <= 0:
            raise ValueError("Некорректные размеры сети или learning_starts")


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 1000
    seed: int = 42
    evaluation_interval: int = 25
    evaluation_episodes: int = 5
    checkpoint_interval: int = 25
    plot_interval: int = 25
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if min(self.episodes, self.evaluation_interval, self.evaluation_episodes,
               self.checkpoint_interval, self.plot_interval, self.torch_threads) <= 0:
            raise ValueError("Число эпизодов, интервалы и потоки должны быть > 0")


@dataclass(frozen=True)
class Config:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
