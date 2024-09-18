import math

import torch
from torch import Tensor


class RandomFourierEmbeddings(torch.nn.Module):
    """Random fourier embeddings for noise levels."""

    def __init__(self, num_channels: int = 32, scale: int = 16, flip_sin_cos: bool = False):
        super().__init__()
        self.register_buffer("frequencies", torch.randn(num_channels // 2) * scale)
        self.register_buffer("pi", torch.tensor(math.pi))
        self.flip_sin_cos = flip_sin_cos

    def forward(self, x: Tensor) -> Tensor:
        x = x * self.frequencies.unsqueeze(0) * 2 * self.pi

        if self.flip_sin_cos:
            out = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        else:
            out = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

        return out


class SinusoidalEmbeddings(torch.nn.Module):
    """Fourier embeddings for noise levels."""

    def __init__(self, num_channels: int = 32, max_period: int = 10000):
        super().__init__()
        zdim = num_channels // 2
        self.register_buffer("frequencies", torch.exp(-math.log(max_period) * torch.arange(0, zdim) / zdim))
        # self.register_buffer('pi', torch.tensor(math.pi))

    def forward(self, x: Tensor) -> Tensor:
        out = x[:] * self.frequencies  # * self.pi
        out = torch.cat((out.sin(), out.cos()), dim=-1)

        return out
