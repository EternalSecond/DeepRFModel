"""Сеть Q(s, a): независимые Q-значения без softmax на выходе."""
from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        if min(state_dim, action_dim, *hidden_sizes) <= 0:
            raise ValueError("Размеры слоев должны быть положительными")
        layers: list[nn.Module] = []
        previous = state_dim
        for size in hidden_sizes:
            layer = nn.Linear(previous, size)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.ReLU()))
            previous = size
        output = nn.Linear(previous, action_dim)
        nn.init.uniform_(output.weight, -0.01, 0.01)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.layers = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)
