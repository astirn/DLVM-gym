from typing import Any, Literal

import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from src.data import ClusteringDataModule, GaussianMixtureDataModule
from src.priors import Prior
from src.priors.crypt.utils import sample_u_initial

from .abstract import DeepLatentVariableModel


class VariationalAutoencoder(DeepLatentVariableModel):

    def __init__(
        self,
        datamodule: ClusteringDataModule | GaussianMixtureDataModule,
        encoder_params: dict[str, Any],
        decoder_params: dict[str, Any],
        prior_params: dict[str, Any],
        kl_warmup_steps: int = 0,
        u_initial_strategy: Literal["data", "stratified data", "random"] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.dim_x = datamodule.dim
        self.num_data = datamodule.num_data
        self.encoder: torch.nn.Module = instantiate(
            encoder_params, dim_x=self.dim_x, **kwargs
        )
        self.decoder: torch.nn.Module = instantiate(
            decoder_params, dim_x=self.dim_x, **kwargs
        )

        if kl_warmup_steps < 0:
            raise ValueError("kl_warmup_steps must be non-negative")
        self.kl_warmup_steps = kl_warmup_steps

        if "crypt" in prior_params["_target_"]:
            if u_initial_strategy in ["data", "stratified data"]:
                u_initial = sample_u_initial(
                    datamodule,
                    k=prior_params["k"],
                    u_keys="x",
                    stratification_key=(
                        "c" if u_initial_strategy == "stratified data" else None
                    ),
                )["x"]
            elif u_initial_strategy == "random":
                u_initial = torch.randn(prior_params["k"], self.dim_x)
            else:
                raise ValueError(
                    f"Invalid u_initial for {prior_params['_target_']}: {u_initial_strategy}"
                )
        else:
            u_initial = None

        self.prior: Prior = instantiate(
            prior_params,
            dim_x=self.dim_x,
            encoder=self.encoder,
            num_data=self.num_data,
            u_initial=u_initial,
            **kwargs,
        )

    def qz(self, *, x: Tensor, **kwargs) -> MultivariateNormal:
        return self.encoder(x)

    def px(self, *, z: Tensor, **kwargs) -> Distribution:
        return self.decoder(z)

    def beta(self, global_step: int) -> float:
        if self.kl_warmup_steps == 0:
            return 1.0
        return min(1.0, global_step / self.kl_warmup_steps)

    def loss(self, batch: dict[str, Tensor], global_step: int) -> Tensor:
        x = batch["x"]
        qz_x = self.qz(x=x)
        z_sample = qz_x.rsample()
        px_z = self.px(z=z_sample)
        beta = self.beta(global_step)
        elbo = px_z.log_prob(x).mean() - beta * self.prior.dkl(qz_x, z_sample)
        return -elbo
