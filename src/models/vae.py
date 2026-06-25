from typing import Any

import torch
from hydra.utils import instantiate
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from src.priors import Prior

from .abstract import DeepLatentVariableModel


class VariationalAutoencoder(DeepLatentVariableModel):

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
        self.prior: Prior = instantiate(prior_params, encoder=self.encoder, **kwargs)

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
