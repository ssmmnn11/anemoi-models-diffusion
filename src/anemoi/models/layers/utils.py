# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

from torch import Tensor
from torch import nn
from torch.utils.checkpoint import checkpoint


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


class AutocastLayerNorm(nn.LayerNorm):
    """LayerNorm that casts the output back to the input type."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward with explicit autocast back to the input type.

        This casts the output to (b)float16 (instead of float32) when we run in mixed
        precision.
        """
        return super().forward(x).type_as(x)


class ConditionalLayerNorm(nn.Module):
    def __init__(
        self, channels: int, noise_emb_dimension: int = 16, w_one_bias_zero_init: bool = True, autocast: bool = True
    ):
        super().__init__()
        # todo: make noise_emb_dimension and init strategy configurable ... how? # out = out * (scale + 1.0) + bias
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)  # no learnable parameters
        self.scale = nn.Linear(noise_emb_dimension, channels)  # , bias=False)
        self.bias = nn.Linear(noise_emb_dimension, channels)  # , bias=False)
        self.autocast = autocast

        if w_one_bias_zero_init:
            nn.init.ones_(self.scale.weight)
            nn.init.zeros_(self.scale.bias)
            nn.init.zeros_(self.bias.weight)
            nn.init.zeros_(self.bias.bias)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        scale = self.scale(emb)
        bias = self.bias(emb)
        out = self.norm(x)
        out = out * (scale + 1.0) + bias
        return out.type_as(x) if self.autocast else out
