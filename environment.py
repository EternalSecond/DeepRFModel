"""Детерминированная 2D-физика с отдельной, лениво создаваемой Pygame-визуализацией.

Современный Gym-подобный контракт: reset -> (state, info), step ->
(state, reward, terminated, truncated, info). Gymnasium не требуется.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from config import EnvironmentConfig

if TYPE_CHECKING:
    import pygame

State = NDArray[np.float32]
Info = dict[str, float | int | bool | str]


class ObstacleEnv:
    """Агент-круг проходит коридор слева направо; препятствия — прямоугольники."""

    def __init__(self, config: EnvironmentConfig | None = None,
                 seed: int = 42) -> None:
        self.config = config or EnvironmentConfig()
        self.rng = np.random.default_rng(seed)
        angles = np.linspace(0, 2 * np.pi, self.config.ray_count, endpoint=False)
        self.directions = np.column_stack((np.cos(angles), np.sin(angles)))
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self.closed = False
        self.obstacles = np.empty((0, 4), dtype=np.float64)
        self.position = np.zeros(2, dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.steps = 0
        self.stale_steps = 0
        self.frontier = 0.0
        self._progress_anchor = 0.0
        self._done = True
        self.reset()

    def reset(self, *, seed: int | None = None) -> tuple[State, Info]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        # Каждая перегородка имеет широкий проход; между перегородками можно
        # свободно двигаться по вертикали. Курс меняется с каждым эпизодом.
        boxes: list[tuple[float, float, float, float]] = [
            (0, 0, c.world_width, c.wall),
            (0, c.height - c.wall, c.world_width, c.height),
            (0, 0, c.wall, c.height),
        ]
        for x in np.linspace(400, c.world_width - 400, c.obstacle_count):
            top = float(self.rng.uniform(c.wall, c.height - c.wall - c.gap_height))
            boxes.extend(((float(x), c.wall, float(x + c.obstacle_width), top),
                          (float(x), top + c.gap_height,
                           float(x + c.obstacle_width), c.height - c.wall)))
        self.obstacles = np.asarray(boxes, dtype=np.float64)
        self.position[:] = (80, c.height / 2)
        self.velocity[:] = 0
        self.frontier = self._progress_anchor = float(self.position[0])
        self.steps = self.stale_steps = 0
        self._done = False
        return self.get_state(), self._info("running")

    def _info(self, reason: str) -> Info:
        return {"reason": reason, "success": reason == "goal", "steps": self.steps,
                "progress": float(np.clip((self.frontier - 80) /
                    (self.config.world_width - 40 - 80), 0, 1))}

    def _collision(self) -> bool:
        # Точное пересечение круга с AABB, включая касание угла.
        closest = np.maximum(self.obstacles[:, :2],
                             np.minimum(self.position, self.obstacles[:, 2:]))
        delta = closest - self.position
        return bool(np.any(np.sum(delta * delta, axis=1) <= self.config.agent_radius ** 2))

    def step(self, action: int) -> tuple[State, float, bool, bool, Info]:
        if self._done:
            raise RuntimeError("Эпизод завершен: вызовите reset()")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 5:
            raise ValueError("action должен быть целым числом от 0 до 4")
        c = self.config
        controls = ((0, 0), (1, 0), (0, -1), (0, 1), (-1, 0))
        acceleration = np.asarray(controls[action]) * c.acceleration
        # Подшаги предотвращают пролет через тонкие препятствия даже при
        # измененных dt/max_speed; время физики не зависит от FPS окна.
        substeps = max(1, math.ceil(c.max_speed * c.dt / (c.agent_radius * 0.5)))
        dt = c.dt / substeps
        collision = False
        for _ in range(substeps):
            self.velocity += (acceleration - c.drag * self.velocity) * dt
            speed = float(np.linalg.norm(self.velocity))
            if speed > c.max_speed:
                self.velocity *= c.max_speed / speed
            self.position += self.velocity * dt
            if self._collision():
                collision = True
                break
        self.steps += 1
        advancement = max(0.0, float(self.position[0]) - self.frontier)
        self.frontier += advancement
        if self.frontier - self._progress_anchor >= 1.0:
            self.stale_steps = 0
            self._progress_anchor = self.frontier
        else:
            self.stale_steps += 1
        # Только новый рекорд x дает награду: движение назад-вперед не фармит ее.
        reward = advancement * c.progress_reward + c.survival_reward
        if advancement < 0.1:
            reward -= c.idle_penalty
        reason = "running"
        terminated = False
        if collision:
            reward -= c.collision_penalty
            reason, terminated = "collision", True
        elif self.position[0] >= c.world_width - 40:
            reward += c.goal_reward
            reason, terminated = "goal", True
        elif self.stale_steps >= c.stagnation_steps:
            reward -= c.stagnation_penalty
            reason, terminated = "stagnation", True
        truncated = not terminated and self.steps >= c.max_steps
        if truncated:
            reason = "time_limit"
        self._done = terminated or truncated
        return self.get_state(), float(reward), terminated, truncated, self._info(reason)

    def raycast(self) -> NDArray[np.float64]:
        """Векторизованный slab test всех лучей против всех прямоугольников."""
        direction = self.directions[:, None, :]
        origin = self.position[None, None, :]
        parallel = np.abs(direction) < 1e-10
        denominator = np.where(parallel, 1.0, direction)
        low = (self.obstacles[None, :, :2] - origin) / denominator
        high = (self.obstacles[None, :, 2:] - origin) / denominator
        near = np.minimum(low, high)
        far = np.maximum(low, high)
        outside = ((origin < self.obstacles[None, :, :2]) |
                   (origin > self.obstacles[None, :, 2:]))
        near = np.where(parallel, -np.inf, near)
        far = np.where(parallel, np.inf, far)
        entry = np.maximum(near.max(axis=2), 0)
        leave = far.min(axis=2)
        valid = (leave >= entry) & ~np.any(parallel & outside, axis=2)
        return np.minimum(np.where(valid, entry, np.inf).min(axis=1),
                          self.config.ray_length)

    def get_state(self) -> State:
        c = self.config
        extra = np.array([
            self.position[0] / c.world_width, self.position[1] / c.height,
            self.velocity[0] / c.max_speed, self.velocity[1] / c.max_speed,
            (self.frontier - self.position[0]) / c.world_width,
            self.stale_steps / c.stagnation_steps, self.steps / c.max_steps,
            (c.world_width - 40 - self.position[0]) / c.world_width,
        ])
        return np.concatenate((self.raycast() / c.ray_length, extra)).astype(np.float32)

    def render(self, *, episode: int = 0, epsilon: float = 0.0,
               reward: float = 0.0, fps: int | None = None) -> bool:
        """Возвращает False при закрытии окна или Escape. Space — пауза."""
        import pygame

        c = self.config
        if self.closed:
            return False
        if self._screen is None:
            pygame.display.init()
            pygame.font.init()
            self._screen = pygame.display.set_mode((c.width, c.height))
            pygame.display.set_caption("DQN obstacle course | Space: pause | Esc: quit")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont("consolas", 18)
        paused = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.close()
                    return False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
            if not paused:
                break
            assert self._clock is not None
            self._clock.tick(30)
        camera = float(np.clip(self.position[0] - c.width * 0.25, 0, c.world_width - c.width))
        screen = self._screen
        screen.fill((17, 22, 32))
        for x1, y1, x2, y2 in self.obstacles:
            pygame.draw.rect(screen, (100, 115, 140), pygame.Rect(round(x1 - camera), round(y1), round(x2 - x1), round(y2 - y1)))
        start = (float(self.position[0] - camera), float(self.position[1]))
        for direction, distance in zip(self.directions, self.raycast()):
            endpoint = self.position + direction * distance
            pygame.draw.line(screen, (40, 110, 105), start, (float(endpoint[0] - camera), float(endpoint[1])), 1)
        pygame.draw.circle(screen, (80, 230, 160), start, round(c.agent_radius))
        goal_x = round(c.world_width - 40 - camera)
        pygame.draw.line(screen, (240, 210, 70), (goal_x, c.wall), (goal_x, c.height - c.wall), 3)
        assert self._font is not None and self._clock is not None
        label = f"Episode {episode} | step {self.steps} | epsilon {epsilon:.3f} | reward {reward:.1f}"
        screen.blit(self._font.render(label, True, (240, 240, 240)), (15, 3))
        pygame.display.flip()
        self._clock.tick(c.fps if fps is None else fps)
        return True

    def close(self) -> None:
        if self._screen is not None:
            import pygame
            pygame.display.quit()
            pygame.font.quit()
        self._screen = None
        self.closed = True
