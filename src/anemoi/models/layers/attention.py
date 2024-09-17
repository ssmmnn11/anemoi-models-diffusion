# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#

import logging
from typing import Optional

import einops
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.transformer import shard_heads
from anemoi.models.distributed.transformer import shard_sequence

LOGGER = logging.getLogger(__name__)


class MultiHeadSelfAttention(nn.Module):
    """Multi Head Self Attention Pytorch Layer."""

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        bias: bool = False,
        is_causal: bool = False,
        window_size: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"

        self.dropout = dropout
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads  # q k v
        self.window_size = (window_size, window_size)  # flash attention
        self.is_causal = is_causal

        self.lin_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)

        self.projection = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attention = self.get_attention_function()

    def get_attention_function(self):
        from torch.nn.functional import scaled_dot_product_attention

        return scaled_dot_product_attention

    def attend(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.attention(query, key, value, is_causal=False)  # expects (batch heads grid variable) format

    def forward(
        self, x: Tensor, shapes: list, batch_size: int, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:
        query, key, value = self.lin_qkv(x).chunk(3, -1)

        if model_comm_group:
            assert (
                model_comm_group.size() == 1 or batch_size == 1
            ), "Only batch size of 1 is supported when model is sharded accross GPUs"

        query, key, value = (
            einops.rearrange(
                t,
                "(batch grid) (heads vars) -> batch heads grid vars",
                batch=batch_size,
                heads=self.num_heads,
            )
            for t in (query, key, value)
        )

        query = shard_heads(query, shapes=shapes, mgroup=model_comm_group)
        key = shard_heads(key, shapes=shapes, mgroup=model_comm_group)
        value = shard_heads(value, shapes=shapes, mgroup=model_comm_group)

        out = self.attend(query, key, value)

        out = shard_sequence(out, shapes=shapes, mgroup=model_comm_group)
        out = einops.rearrange(out, "batch heads grid vars -> (batch grid) (heads vars)")

        out = self.projection(out)

        return out


class FlashMultiHeadSelfAttention(MultiHeadSelfAttention):
    """Multi Head Self Attention Pytorch Layer."""

    def get_attention_function(self):
        from flash_attn import flash_attn_func

        return flash_attn_func

    def attend(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:

        query, key, value = (
            einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
        )

        out = self.attention(query, key, value, causal=False, window_size=self.window_size)
        out = einops.rearrange(out, "batch grid heads vars -> batch heads grid vars")

        return out
