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
        encoder_params: dict[str, Any],
        decoder_params: dict[str, Any],
        prior_params: dict[str, Any],
        **kwargs,
    ):
        super().__init__()
        self.encoder: torch.nn.Module = instantiate(encoder_params, **kwargs)
        self.decoder: torch.nn.Module = instantiate(decoder_params, **kwargs)
        self.prior: PriorTypes = instantiate(
            prior_params, encoder=self.encoder, **kwargs
        )

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
