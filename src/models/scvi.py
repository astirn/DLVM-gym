from typing import Any, Literal

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from src.data.scRNA import scRNABatch
from src.priors import Prior

from .abstract import DeepLatentVariableModel


class scVI(DeepLatentVariableModel):

    def __init__(
        self,
        num_batches: int,
        encoder_params: dict[str, Any],
        decoder_params: dict[str, Any],
        prior_params: dict[str, Any],
        **kwargs,
    ):
        super().__init__()
        self.num_batches = num_batches
        self.encoder: torch.nn.Module = instantiate(encoder_params, **kwargs)
        self.decoder: torch.nn.Module = instantiate(decoder_params, **kwargs)
        self.prior: Prior = instantiate(prior_params, encoder=self.encoder, **kwargs)

    def qz(self, x: Tensor, s: Tensor) -> MultivariateNormal:
        s = F.one_hot(s, num_classes=self.num_batches).squeeze(1).float()
        return self.encoder(x, s)

    def px(self, z: Tensor, s: Tensor) -> Distribution:
        s = F.one_hot(s, num_classes=self.num_batches).squeeze(1).float()
        return self.decoder(z, s)

    def local_distributions(self, batch: scRNABatch) -> dict[str, Distribution]:
        qz_x = self.qz(batch.x, batch.s)
        z = qz_x.rsample()
        px_z = self.px(z, batch.s)
        return {**{"qz_x": qz_x, "px_z": px_z}, **self.prior.local_distributions(qz_x)}

    def loss(self, batch: scRNABatch) -> Tensor:
        qz_x, px_z = tuple(self.local_distributions(batch).values())[:2]
        elbo = px_z.log_prob(batch.x).mean() - self.prior.dkl(qz_x)
        return -elbo
