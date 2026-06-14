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

    def qz(self, x: Tensor) -> MultivariateNormal:
        return self.encoder(x)

    def px(self, z: Tensor) -> Distribution:
        return self.decoder(z)

    def local_distributions(self, batch: dict[str, Tensor]) -> dict[str, Distribution]:
        qz_x = self.qz(batch["x"])
        z = qz_x.rsample()
        px_z = self.px(z)
        return {**{"qz_x": qz_x, "px_z": px_z}, **self.prior.local_distributions(qz_x)}

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        qz_x, px_z = tuple(self.local_distributions(batch).values())[:2]
        elbo = px_z.log_prob(batch["x"]).mean() - self.prior.dkl(qz_x)
        return -elbo
