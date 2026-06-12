from typing import Any

import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from src.priors import PriorTypes

from .wrapper import ModelProtocol


class VariationalAutoencoder(torch.nn.Module, ModelProtocol):

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        encoder_params: dict[str, Any],
        decoder_params: dict[str, Any],
        prior: PriorTypes,
        **kwargs,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.encoder = instantiate(encoder_params, dim_x=dim_x, dim_z=dim_z)
        self.decoder = instantiate(decoder_params, dim_x=dim_x, dim_z=dim_z)
        self.prior = prior

    def qz(self, x: Tensor) -> MultivariateNormal:
        return self.encoder(x)

    def px(self, z: Tensor) -> Distribution:
        return self.decoder(z)

    def loss(self, batch: dict[str, Tensor], update_metrics: bool = False) -> Tensor:
        x = batch["x"]
        qz_x = self.qz(x)
        z = qz_x.rsample()
        elbo = self.px(z).log_prob(x).mean() - self.prior.dkl(qz_x)

        if update_metrics:
            pass

        return -elbo

    def metrics(self) -> dict[str, float]:
        return {}
