# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
import einops
import uuid
from typing import Optional
from typing import Tuple


import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.preprocessing import Processors


class AnemoiModelInterface(torch.nn.Module):
    """An interface for Anemoi models.

    This class is a wrapper around the Anemoi model that includes pre-processing and post-processing steps.
    It inherits from the PyTorch Module class.

    Attributes
    ----------
    config : DotDict
        Configuration settings for the model.
    id : str
        A unique identifier for the model instance.
    multi_step : bool
        Whether the model uses multi-step input.
    graph_data : HeteroData
        Graph data for the model.
    statistics : dict
        Statistics for the data.
    metadata : dict
        Metadata for the model.
    data_indices : dict
        Indices for the data.
    model : AnemoiModelEncProcDec
        The underlying Anemoi model.
    """

    def __init__(
        self,
        *,
        config: DotDict,
        graph_data: HeteroData,
        statistics: dict,
        data_indices: dict,
        metadata: dict,
        statistics_tendencies: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.id = str(uuid.uuid4())
        self.multi_step = self.config.training.multistep_input
        self.prediction_strategy = self.config.training.prediction_strategy
        self.graph_data = graph_data
        self.statistics = statistics
        self.statistics_tendencies = statistics_tendencies
        self.metadata = metadata
        self.data_indices = data_indices

        # tood : from config
        self.sigma_max = config.training.noise.sigma_max
        self.sigma_min = config.training.noise.sigma_min
        self.sigma_data = config.training.noise.sigma_data

        self._build_model()

    def _build_model(self) -> None:
        """Builds the model and pre- and post-processors."""
        # Instantiate processors for state
        processors_state = [
            [name, instantiate(processor, statistics=self.statistics, data_indices=self.data_indices)]
            for name, processor in self.config.data.processors.state.items()
        ]

        # Assign the processor list pre- and post-processors
        self.pre_processors_state = Processors(processors_state)
        self.post_processors_state = Processors(processors_state, inverse=True)

        # Instantiate processors for tendency
        self.pre_processors_tendency = None
        self.post_processors_tendency = None
        if self.prediction_strategy == "tendency":
            processors_tendency = [
                [name, instantiate(processor, statistics=self.statistics_tendencies, data_indices=self.data_indices)]
                for name, processor in self.config.data.processors.tendency.items()
            ]

            self.pre_processors_tendency = Processors(processors_tendency)
            self.post_processors_tendency = Processors(processors_tendency, inverse=True)

        # Instantiate the model
        self.model = instantiate(
            self.config.model.model,
            model_config=self.config,
            data_indices=self.data_indices,
            graph_data=self.graph_data,
            _recursive_=False,  # Disables recursive instantiation by Hydra
        )

    def forward(
        self,
        x: torch.Tensor,
        state_in: torch.Tensor,
        sigma: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> torch.Tensor:
        if self.prediction_strategy == "residual":

            assert 1 > 2, "Not implemented, for diffusion model"

            # Predict state by adding residual connection (just for the prognostic variables)
            x_pred = self.model.forward(x, model_comm_group)
            x_pred[..., self.model._internal_output_idx] += x[:, -1, :, :, self.model._internal_input_idx]
        else:
            x_pred = self.model.forward(x, state_in, sigma, model_comm_group)

        for bounding in self.model.boundings:
            # bounding performed in the order specified in the config file
            x_pred = bounding(x_pred)

        return x_pred

    def predict_step(self, batch: torch.Tensor, fcstep: int = 0, plot_diag: bool = False) -> torch.Tensor:
        """Prediction step for the model.

        Parameters
        ----------
        batch : torch.Tensor
            Input batched data.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """

        with torch.no_grad():

            # assert (
            #     len(batch.shape) == 1
            # ), f"The input tensor has an incorrect shape: expected a 4-dimensional tensor, got {batch.shape}!"
            x = self.pre_processors_state(batch[:, 0 : self.multi_step, ...], in_place=False)

            # Dimensions are
            # batch, timesteps, horizontal space, variables
            #x = x[..., None, :, :]  # add dummy ensemble dimension as 3rd index
            # extra_args = {}
            extra_args = {"S_churn": 2.5, "S_min": 0.75, "S_max": float("inf"), "S_noise": 1.05}
            noise_steps = self.noise_schedule(
                device=x.device, dtype=torch.float64, nsteps=20, sigma_min=0.002, sigma_max=80, rho=7
            )
            
            if self.prediction_strategy == "tendency":
                tendency_hat = self.default_sampler(x, noise_steps, plot_diag=plot_diag, **extra_args)
                y_hat = self.add_tendency_to_state(x[:, -1, ...], tendency_hat)
            else:
                y_hat = self(x)
                y_hat = self.post_processors_state(y_hat, in_place=False)

        return y_hat

    def add_tendency_to_state(self, state_inp: torch.Tensor, tendency: torch.Tensor) -> torch.Tensor:
        """Add the tendency to the state.

        Parameters
        ----------
        state_inp : torch.Tensor
            The input state tensor with full input variables and unprocessed.
        tendency : torch.Tensor
            The tendency tensor output from model.

        Returns
        -------
        torch.Tensor
            Predicted data.
        """

        state_outp = self.post_processors_tendency(
            tendency, in_place=False, data_index=self.data_indices.data.output.full
        )

        state_outp[..., self.data_indices.model.output.diagnostic] = self.post_processors_state(
            tendency[..., self.data_indices.model.output.diagnostic],
            in_place=False,
            data_index=self.data_indices.data.output.diagnostic,
        )

        state_outp[..., self.data_indices.model.output.prognostic] += state_inp[
            ..., self.data_indices.model.input.prognostic
        ]

        return state_outp

    def compute_tendency(self, x_t1: torch.Tensor, x_t0: torch.Tensor) -> torch.Tensor:
        tendency = self.pre_processors_tendency(
            x_t1[..., self.data_indices.data.output.full] - x_t0[..., self.data_indices.data.output.full],
            in_place=False,
            data_index=self.data_indices.data.output.full,
        )
        # diagnostic variables are taken from x_t1, normalised as full fields:
        tendency[..., self.data_indices.model.output.diagnostic] = self.pre_processors_state(
            x_t1[..., self.data_indices.data.output.diagnostic],
            in_place=False,
            data_index=self.data_indices.data.output.diagnostic,
        )
        return tendency


    def noise_schedule(self, device, dtype=None, nsteps=20, sigma_min=0.002, sigma_max=80, rho=7):
        if dtype is None:
            steps_idx = torch.arange(nsteps, device=device)
        else:
            steps_idx = torch.arange(nsteps, device=device, dtype=dtype)
        noise_steps = (sigma_max ** (1 / rho) + steps_idx / (nsteps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        noise_steps = torch.cat([torch.as_tensor(noise_steps), torch.zeros_like(noise_steps[:1])])

        return noise_steps
    
    def default_sampler(
        self,
        x_in,
        noise_steps,
        plot_diag=False,
        S_churn=0,  # 2.5
        S_min=0,  # 0.75
        S_max=float("inf"),
        S_noise=1.0,  # 1,  # 1.05
        dtype=torch.float64,
    ):
        batch_size = x_in.shape[0]
        assert batch_size == 1, "need to adapt for bs > 1 ?"

        nsteps = len(noise_steps) - 1
        x_next = torch.randn_like(x_in[:, 0, ..., : self.model.num_output_channels]) * noise_steps[0]

        for i, (t_cur, t_next) in enumerate(zip(noise_steps[:-1], noise_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / nsteps, 2 ** (1.0 / 2.0) - 1) if S_min <= t_cur <= S_max else 0

            t_hat = t_cur + gamma * t_cur
            t_hat = t_hat.view(1, 1, 1, 1)
            t_next = t_next.view(1, 1, 1, 1)
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)
            
            # Euler step.
            denoised = self.fwd_with_preconditioning(x_hat.to(dtype=x_in.dtype), t_hat.to(x_in.dtype), x_in).to(
                dtype
            )

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < nsteps - 1:
                denoised = self.fwd_with_preconditioning(
                    x_next.to(dtype=x_in.dtype), t_next.to(x_in.dtype), x_in
                ).to(dtype)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

            if plot_diag:
                self._plot_denoising(denoised, x_next, i)

        return x_next
    def fwd_with_preconditioning(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        state_in: torch.Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> torch.Tensor:
        sigma_data = self.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        c_in = 1.0 / (sigma_data**2 + sigma**2) ** 0.5
        c_noise = sigma.log() / 4.0  # used to condition on noise levels

        pred = self((c_in * x), state_in, c_noise, model_comm_group=model_comm_group)

        D_x = c_skip * x + c_out * pred

        return D_x
