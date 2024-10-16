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
import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.shapes import get_shape_shards
from anemoi.models.layers.graph import TrainableTensor
from anemoi.models.layers.mlp import MLP1
from anemoi.models.layers.noise import SinusoidalEmbeddings

LOGGER = logging.getLogger(__name__)


class AnemoiModelEncProcDec(nn.Module):
    """Message passing graph neural network."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: dict,
        graph_data: HeteroData,
    ) -> None:
        """Initializes the graph neural network.

        Parameters
        ----------
        model_config : DotDict
            Model configuration
        data_indices : dict
            Data indices
        graph_data : HeteroData
            Graph definition
        """
        super().__init__()

        self._graph_data = graph_data
        self._graph_name_data = model_config.graph.data
        self._graph_name_hidden = model_config.graph.hidden

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        self.multi_step = model_config.training.multistep_input

        self.encoder_grad_checkpointing = model_config.model.encoder_grad_checkpointing
        self.decoder_grad_checkpointing = model_config.model.decoder_grad_checkpointing

        self._define_tensor_sizes(model_config)

        # Create trainable tensors
        self._create_trainable_attributes()

        # Register lat/lon of nodes
        self._register_latlon("data", self._graph_name_data)
        self._register_latlon("hidden", self._graph_name_hidden)

        self.data_indices = data_indices

        self.num_channels = model_config.model.num_channels

        input_dim = (
            self.multi_step * self.num_input_channels
            + self.num_output_channels
            + self.latlons_data.shape[1]
            + self.trainable_data_size
        )

        # Encoder data -> hidden
        self.encoder = instantiate(
            model_config.model.encoder,
            in_channels_src=input_dim,
            in_channels_dst=self.latlons_hidden.shape[1] + self.trainable_hidden_size,
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)],
            src_grid_size=self._data_grid_size,
            dst_grid_size=self._hidden_grid_size,
        )

        # Processor hidden -> hidden
        self.processor = instantiate(
            model_config.model.processor,
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self._hidden_grid_size,
            dst_grid_size=self._hidden_grid_size,
        )

        # Decoder hidden -> data
        self.decoder = instantiate(
            model_config.model.decoder,
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim,
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)],
            src_grid_size=self._hidden_grid_size,
            dst_grid_size=self._data_grid_size,
        )

        self.noise_embedder = SinusoidalEmbeddings(num_channels=32, max_period=1000)

        self.noise_mlp = MLP1(
            in_features=32, hidden_dim=32, out_features=16, n_extra_layers=-1, final_activation=False, layer_norm=False
        )

        # Instantiation of model output bounding functions (e.g., to ensure outputs like TP are positive definite)
        self.boundings = nn.ModuleList(
            [
                instantiate(cfg, name_to_index=self.data_indices.model.output.name_to_index)
                for cfg in getattr(model_config.model, "bounding", [])
            ]
        )

    def _calculate_shapes_and_indices(self, data_indices: dict) -> None:
        self.num_input_channels = len(data_indices.internal_model.input)
        self.num_output_channels = len(data_indices.internal_model.output)
        self._internal_input_idx = data_indices.internal_model.input.prognostic
        self._internal_output_idx = data_indices.internal_model.output.prognostic

    def _assert_matching_indices(self, data_indices: dict) -> None:

        assert len(self._internal_output_idx) == len(data_indices.internal_model.output.full) - len(
            data_indices.internal_model.output.diagnostic
        ), (
            f"Mismatch between the internal data indices ({len(self._internal_output_idx)}) and "
            f"the internal output indices excluding diagnostic variables "
            f"({len(data_indices.internal_model.output.full) - len(data_indices.internal_model.output.diagnostic)})",
        )
        assert len(self._internal_input_idx) == len(
            self._internal_output_idx,
        ), f"Internal model indices must match {self._internal_input_idx} != {self._internal_output_idx}"

    def _define_tensor_sizes(self, config: DotDict) -> None:
        self._data_grid_size = self._graph_data[self._graph_name_data].num_nodes
        self._hidden_grid_size = self._graph_data[self._graph_name_hidden].num_nodes

        self.trainable_data_size = config.model.trainable_parameters.data
        self.trainable_hidden_size = config.model.trainable_parameters.hidden

    def _register_latlon(self, name: str, nodes: str) -> None:
        """Register lat/lon buffers.

        Parameters
        ----------
        name : str
            Name to store the lat-lon coordinates of the nodes.
        nodes : str
            Name of nodes to map
        """
        coords = self._graph_data[nodes].x
        sin_cos_coords = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        self.register_buffer(f"latlons_{name}", sin_cos_coords, persistent=True)

    def _create_trainable_attributes(self) -> None:
        """Create all trainable attributes."""
        self.trainable_data = TrainableTensor(trainable_size=self.trainable_data_size, tensor_size=self._data_grid_size)
        self.trainable_hidden = TrainableTensor(
            trainable_size=self.trainable_hidden_size, tensor_size=self._hidden_grid_size
        )

    def _run_mapper(
        self,
        mapper: nn.Module,
        data: tuple[Tensor],
        noise_levels: tuple[Tensor],
        batch_size: int,
        shard_shapes: tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        use_grad_checkpoint: bool = False,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Run mapper with activation checkpoint.

        Parameters
        ----------
        mapper : nn.Module
            Which processor to use
        data : tuple[Tensor]
            tuple of data to pass in
        noise_levels : tuple[Tensor],
            tuple of noise levels
        batch_size: int,
            Batch size
        shard_shapes : tuple[tuple[int, int], tuple[int, int]]
            Shard shapes for the data
        model_comm_group : ProcessGroup
            model communication group, specifies which GPUs work together
            in one model instance
        use_grad_checkpoint : bool, optional
            Use gradient checkpointing, by default False
        use_reentrant : bool, optional
            Use reentrant, by default False

        Returns
        -------
        Tensor
            Mapped data
        """
        if use_grad_checkpoint:
            return checkpoint(
                mapper,
                data,
                batch_size=batch_size,
                noise_levels=noise_levels,
                shard_shapes=shard_shapes,
                model_comm_group=model_comm_group,
                use_reentrant=use_reentrant,
            )
        else:
            return mapper(
                data,
                batch_size=batch_size,
                noise_levels=noise_levels,
                shard_shapes=shard_shapes,
                model_comm_group=model_comm_group,
            )

    def make_noise_emb(self, noise_emb: Tensor, repeat: int) -> Tensor:
        out = einops.repeat(
            noise_emb, "batch ensemble noise_level vars -> batch ensemble (repeat noise_level) vars", repeat=repeat
        )
        out = einops.rearrange(out, "batch ensemble grid vars -> (batch ensemble grid) vars")
        return out

    def forward(
        self, x: Tensor, state_in: Tensor, noise: Tensor, model_comm_group: Optional[ProcessGroup] = None
    ) -> Tensor:
        batch_size = state_in.shape[0]
        ensemble_size = state_in.shape[2]

        # combine noised target, input state and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch ensemble grid vars -> (batch ensemble grid) vars"),
                einops.rearrange(state_in, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                self.trainable_data(self.latlons_data, batch_size=batch_size),
            ),
            dim=-1,  # feature dimension
        )

        x_hidden_latent = self.trainable_hidden(self.latlons_hidden, batch_size=batch_size)

        noise_emb = self.noise_embedder(noise)
        noise_emb = self.noise_mlp(noise_emb)

        # these are created once, we need these because we need to keep track what noise belongs to what batch
        # in the conditional layer norms ...

        noise_data = self.make_noise_emb(noise_emb, repeat=self._data_grid_size)
        noise_hidden = self.make_noise_emb(noise_emb, repeat=self._hidden_grid_size)
        noise_data_to_hidden = self.make_noise_emb(
            noise_emb,
            repeat=self._graph_data[(self._graph_name_data, "to", self._graph_name_hidden)]["edge_length"].shape[0],
        )
        noise_hidden_to_data = self.make_noise_emb(
            noise_emb,
            repeat=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_data)]["edge_length"].shape[0],
        )
        # only GNNProcessor and GNNTransformerProcessor support noise_hidden_to_hidden
        # noise_hidden_to_hidden = self.make_noise_emb(
        #     noise_emb,
        #     repeat=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)]["edge_length"]
        # )

        # get shard shapes
        shard_shapes_data = get_shape_shards(x_data_latent, 0, model_comm_group)
        shard_shapes_hidden = get_shape_shards(x_hidden_latent, 0, model_comm_group)

        shape_noise_data = get_shape_shards(noise_data, 0, model_comm_group)
        shape_noise_hidden = get_shape_shards(noise_hidden, 0, model_comm_group)

        # Run encoder
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            noise_levels=(noise_data, noise_hidden, noise_data_to_hidden),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden, shape_noise_data, shape_noise_hidden),
            model_comm_group=model_comm_group,
            use_grad_checkpoint=self.encoder_grad_checkpointing,
        )

        x_latent_proc = self.processor(
            x_latent,
            noise_levels=noise_hidden,  # todo: make it work with GNNprocessor and GNNTransformerProcessor
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shape_noise_hidden),
            model_comm_group=model_comm_group,
        )

        # add skip connection (hidden -> hidden)
        x_latent_proc = x_latent_proc + x_latent

        # Run decoder
        x_out = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            noise_levels=(noise_hidden, noise_data, noise_hidden_to_data),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data, shape_noise_hidden, shape_noise_data),
            model_comm_group=model_comm_group,
            use_grad_checkpoint=self.decoder_grad_checkpointing,
        )

        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .to(dtype=x.dtype)
            .clone()
        )

        return x_out
