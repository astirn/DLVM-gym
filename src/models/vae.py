from typing import Any

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
        stratified_u_initial: bool = False,
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

        u_initial = sample_u_initial(
            datamodule,
            k=prior_params["k"],
            u_keys="x",
            stratification_key="c" if stratified_u_initial else None,
        )["x"]

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

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        x = batch["x"]
        qz_x = self.qz(x=x)
        z_sample = qz_x.rsample()
        px_z = self.px(z=z_sample)
        elbo = px_z.log_prob(x).mean() - self.prior.dkl(qz_x, z_sample)
        return -elbo
