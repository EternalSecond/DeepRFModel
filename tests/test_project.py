"""Проверки геометрии, семантики Bellman и реального цикла обучения."""
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

from agent import DQNAgent, ReplayBuffer
from config import AgentConfig, Config, EnvironmentConfig, TrainingConfig
from environment import ObstacleEnv
from trainer import Trainer


class EnvironmentTests(unittest.TestCase):
    def test_seed_and_observation(self) -> None:
        a, b = ObstacleEnv(seed=7), ObstacleEnv(seed=7)
        np.testing.assert_array_equal(a.obstacles, b.obstacles)
        for action in [1, 2, 1, 3, 0]:
            sa, ra, ta, xa, _ = a.step(action)
            sb, rb, tb, xb, _ = b.step(action)
            np.testing.assert_array_equal(sa, sb)
            self.assertEqual((ra, ta, xa), (rb, tb, xb))
        self.assertEqual(sa.shape, (a.config.state_dim,))
        self.assertEqual(sa.dtype, np.float32)
        self.assertTrue(np.isfinite(sa).all())

    def test_rays_parallel_and_exact_distance(self) -> None:
        env = ObstacleEnv(replace(EnvironmentConfig(), ray_count=4))
        env.position[:] = (100, 100)
        env.obstacles = np.array([[150, 90, 160, 110], [90, 150, 110, 160]], dtype=float)
        np.testing.assert_allclose(env.raycast(), [50, 50, 400, 400])

    def test_collision_and_reset_guard(self) -> None:
        env = ObstacleEnv()
        env.position[:] = (env.config.wall + env.config.agent_radius + 0.1, 100)
        _, reward, terminal, truncated, info = env.step(4)
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        self.assertLess(reward, -20)
        self.assertEqual(info["reason"], "collision")
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_no_tunneling(self) -> None:
        env = ObstacleEnv(replace(EnvironmentConfig(), dt=0.2, max_speed=2000, drag=0))
        env.obstacles = np.array([[150, 0, 151, 600]], dtype=float)
        env.position[:] = (80, 300)
        env.velocity[:] = (2000, 0)
        self.assertTrue(env.step(0)[2])
        self.assertLess(env.position[0], 151)

    def test_stagnation_and_time_limit_differ(self) -> None:
        env = ObstacleEnv(replace(EnvironmentConfig(), stagnation_steps=2))
        env.step(0)
        _, _, terminal, truncated, info = env.step(0)
        self.assertTrue(terminal)
        self.assertFalse(truncated)
        self.assertEqual(info["reason"], "stagnation")
        env = ObstacleEnv(replace(EnvironmentConfig(), max_steps=1))
        _, _, terminal, truncated, _ = env.step(1)
        self.assertFalse(terminal)
        self.assertTrue(truncated)

    def test_goal_and_frontier_reward(self) -> None:
        env = ObstacleEnv()
        env.position[:] = (env.config.world_width - 41, 300)
        env.frontier = env._progress_anchor = env.position[0]
        env.velocity[:] = (100, 0)
        _, reward, terminal, _, info = env.step(1)
        self.assertTrue(terminal)
        self.assertTrue(info["success"])
        self.assertGreater(reward, 99)
        env.reset()
        env.frontier = 200
        self.assertLess(env.step(1)[1], 0)

    def test_pygame_render_and_close(self) -> None:
        import pygame
        env = ObstacleEnv()
        self.assertTrue(env.render(fps=0))
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        self.assertFalse(env.render(fps=0))
        self.assertTrue(env.closed)


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.config = AgentConfig(batch_size=2, replay_capacity=8, learning_starts=2,
                                  hidden_sizes=(8,), train_frequency=1, target_update_interval=2)

    def test_ring_buffer_copies_states(self) -> None:
        replay = ReplayBuffer(2, 3)
        state = np.zeros(3, dtype=np.float32)
        replay.add(state, 1, 1, state, False)
        state[:] = 9
        self.assertEqual(replay.states[0, 0], 0)
        replay.add(state, 2, 2, state, True)
        replay.add(state, 3, 3, state, False)
        self.assertEqual(len(replay), 2)
        self.assertEqual(set(replay.actions), {2, 3})

    def test_bellman_terminal_mask(self) -> None:
        config = replace(self.config, gamma=0.5)
        agent = DQNAgent(3, 2, config)
        for parameter in agent.online.parameters():
            parameter.data.zero_()
        for parameter in agent.target.parameters():
            parameter.data.zero_()
        agent.target.layers[-1].bias.data.fill_(4)
        state = np.zeros(3, dtype=np.float32)
        agent.observe(state, 0, 1, state, True)   # target = 1 -> Huber 0.5
        agent.observe(state, 0, 1, state, False)  # target = 3 -> Huber 2.5
        self.assertAlmostEqual(agent.optimize(), 1.5, places=5)

    def test_target_sync_and_checkpoint_continuation(self) -> None:
        agent = DQNAgent(3, 2, self.config)
        state = np.ones(3, dtype=np.float32)
        for _ in range(4):
            agent.observe(state, 1, 3, state, False)
        old = [p.clone() for p in agent.target.parameters()]
        self.assertIsNotNone(agent.optimize())
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old, agent.target.parameters())))
        agent.optimize()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            agent.save(path, metadata={"test": 1})
            restored = DQNAgent(3, 2, self.config, seed=999)
            self.assertEqual(restored.load(path), {"test": 1})
            self.assertEqual(agent.epsilon, restored.epsilon)
            self.assertEqual(agent.act(state), restored.act(state))
            self.assertAlmostEqual(agent.optimize(), restored.optimize(), places=6)
            for a, b in zip(agent.online.parameters(), restored.online.parameters()):
                torch.testing.assert_close(a, b)

    def test_training_and_resume(self) -> None:
        config = Config(replace(EnvironmentConfig(), max_steps=8), self.config,
                        TrainingConfig(episodes=2, evaluation_interval=1,
                                       evaluation_episodes=1, plot_interval=2))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trainer = Trainer(config, output)
            trainer.train()
            self.assertGreater(trainer.agent.optimization_steps, 0)
            for name in ("best_model.pth", "last_checkpoint.pth", "metrics.csv", "training_progress.png"):
                self.assertTrue((output / name).is_file())
            resumed = Trainer(replace(config, training=replace(config.training, episodes=3)), output)
            resumed.resume(output / "last_checkpoint.pth")
            resumed.train()
            self.assertEqual(resumed.completed_episodes, 3)
            self.assertEqual(len(resumed.history), 3)


if __name__ == "__main__":
    unittest.main()
