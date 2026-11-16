# DeepRFModel

A Double DQN agent learning to navigate a procedural 2D obstacle course with PyTorch and Pygame.

## Features

- Fixed-step physics, collision substeps and 16 raycasting sensors.
- Double DQN (or standard DQN), experience replay and a target network.
- Epsilon-greedy exploration, Huber loss and gradient clipping.
- Headless training, interactive visualization and reproducible course seeds.
- Evaluation, resumable checkpoints, CSV metrics and Matplotlib charts.

## Quick start

Requires Python 3.11 or newer.

```bash
git clone https://github.com/EternalSecond/DeepRFModel.git
cd DeepRFModel
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python trainer.py --episodes 1000 --render
```

Omit `--render` for faster training without a window. Space pauses the simulation;
Escape or closing the window stops training and saves a checkpoint. Ctrl+C also saves.
Use `--device cpu` to select CPU explicitly. CUDA requires a compatible PyTorch build.

## Train, resume and evaluate

```bash
python trainer.py --episodes 1000 --output runs/experiment1
python trainer.py --resume runs/experiment1/last_checkpoint.pth --episodes 2000 --output runs/experiment1
python trainer.py --evaluate runs/experiment1/best_model.pth --episodes 10 --render
python trainer.py --evaluate runs/experiment1/best_model.pth --seed 12345 --episodes 100
python -m unittest discover -s tests -v
```

`--episodes` is the total training episode count, including completed episodes on resume.
For evaluation it selects the number of evaluation episodes. Use a new output directory
for each experiment: existing results in that directory can be overwritten.

Edit `config.py` or pass `--config path/to/config.json` to change settings. The trainer
writes a complete JSON configuration into the output directory. Resume automatically
reads configuration from the checkpoint.

## Project structure

| Module | Responsibility |
| --- | --- |
| `config.py` | Validated configuration dataclasses |
| `environment.py` | Physics, procedural courses, sensors, rewards and rendering |
| `model.py` | MLP: 24 inputs, two 256-unit ReLU layers, 5 Q-values |
| `agent.py` | Replay buffer, Double DQN updates and checkpoint persistence |
| `trainer.py` | Training CLI, evaluation, metrics and charts |
| `tests/test_project.py` | Geometry, Bellman targets, checkpoints and integration tests |

## Environment API

```python
from environment import ObstacleEnv

env = ObstacleEnv(seed=42)
state, info = env.reset(seed=42)
next_state, reward, terminated, truncated, info = env.step(1)
env.close()
```

The API follows the modern Gym convention but does not subclass `gymnasium.Env`
or expose Gym spaces. Reset after termination or truncation.

Actions: `0` coast, `1` right, `2` up, `3` down, `4` left. Controls apply acceleration;
drag slows the agent. Physics uses a fixed 1/30-second step independently of rendering.

The float32 observation contains 16 normalized ray distances followed by position,
velocity, distance behind the furthest reached position, stagnation timer, episode time
and remaining goal distance. Rays start facing right and rotate clockwise in screen
coordinates, with a maximum range of 400 pixels. Hidden course geometry makes this
partially observable.

Rewards: +0.08 per pixel of new horizontal progress, +0.002 per step, -0.03 for
insufficient progress, -30 for collision, -15 for stagnation and +100 for reaching
the goal. Backtracking cannot earn the same progress reward twice. Collision, goal
and stagnation terminate the episode; the step limit truncates it.

## Learning defaults

Adam learning rate 0.0003; discount 0.99; batch size 128; replay capacity 100,000.
Learning starts after 2,000 transitions and runs every four interactions. Epsilon
decays exponentially from 1.0 toward 0.05 with an 80,000-step scale. The target
network synchronizes every 1,000 optimizer updates.

```text
y = reward + gamma * (1 - terminated) * Q_target(next_state, argmax Q_online(next_state))
loss = SmoothL1Loss(Q_online(state, action), y)
```

Time-limit truncations retain bootstrapping from the final observation before reset.
Set `double_dqn` to `false` in the agent configuration for standard DQN.

## Outputs

- `best_model.pth`: best policy by mean greedy evaluation reward; excludes replay.
- `last_checkpoint.pth`: networks, optimizer, replay, RNG states, counters and metrics.
- `metrics.csv`: episode reward, loss, epsilon, progress, termination reason and evaluation results.
- `training_progress.png`: reward, moving average, loss, epsilon and progress.
- `config.json`: experiment configuration.

Evaluation runs every 25 episodes and at the end on fixed courses without exploration.
Use a different seed for final testing and inspect success rate alongside reward.
Only `last_checkpoint.pth` supports training resume. Checkpoints are written through
atomic replacement and loaded with `weights_only=True`; use trusted files.

## Limitations and reproducibility

This is a runnable experimentation project. Successful navigation and learning convergence
are not guaranteed by a specific episode count. Automated tests verify functionality,
not policy quality. No pretrained model is included.

Seeds support repeatable runs on the same setup; identical results across PyTorch
versions, devices or CUDA configurations are not guaranteed. After interruption during
an episode, replay is preserved but the incomplete episode restarts on resume and its
metrics are excluded.

Virtual environments, IDE settings, credentials and local training artifacts are excluded
from publication. Do not commit private datasets or secrets when extending the project.
