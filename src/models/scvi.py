from typing import Any, Literal

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from src.priors import Prior

from .abstract import DeepLatentVariableModel


class scVI(DeepLatentVariableModel):

    def __init__(
        self,
        encoder_params: dict[str, Any],
        decoder_params: dict[str, Any],
        prior_params: dict[str, Any],
        **kwargs,
    ):
        super().__init__()
        self.num_batches = kwargs["num_batches"]
        self.num_genes = kwargs["num_genes"]
        self.encoder: torch.nn.Module = instantiate(encoder_params, **kwargs)
        self.decoder: torch.nn.Module = instantiate(decoder_params, **kwargs)
        self.prior: Prior = instantiate(
            prior_params,
            dim_x=kwargs["num_genes"] + self.num_batches,
            encoder=self.encoder,
            u_constraint=self.u_constraint,
            **kwargs,
        )

    def u_constraint(self, u: Tensor) -> Tensor:
        x = u[:, : self.num_genes]
        s = u[:, self.num_genes :]
        return torch.concat([F.softplus(x), F.softmax(s, dim=-1)], dim=1)

    def qz(self, x: Tensor, s: Tensor) -> MultivariateNormal:
        xs = torch.concat([x, s], dim=1)
        return self.encoder(xs)

    def px(self, z: Tensor, s: Tensor, l: Tensor) -> Distribution:
        zs = torch.concat([z, s], dim=1)
        return self.decoder(zs, l)

    def local_distributions(self, batch: dict[str, Tensor]) -> dict[str, Distribution]:
        s = F.one_hot(batch["s"], num_classes=self.num_batches).float()
        qz_xs = self.qz(batch["x"].log1p(), s)
        z = qz_xs.rsample()
        px_zs = self.px(z, s, batch["x"].sum(dim=1, keepdim=True))
        return {
            **{"qz_xs": qz_xs, "px_zs": px_zs},
            **self.prior.local_distributions(qz_xs),
        }

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        qz_xs, px_zs = tuple(self.local_distributions(batch).values())[:2]
        elbo = px_zs.log_prob(batch["x"]).mean() - self.prior.dkl(qz_xs)
        return -elbo
