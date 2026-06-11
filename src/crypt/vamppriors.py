import math

import torch
import torch.distributions as dist
from torch import Tensor

from .abstract import AbstractVampPrior


class VampPrior(AbstractVampPrior):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_pz_c(self, z: Tensor) -> Tensor:
        pz_c = self.qz(self.u)
        return pz_c.log_prob(z[:, None])

    def qc(self, log_pz_c: Tensor) -> dist.Categorical:
        return dist.Categorical(logits=log_pz_c)

    def dkl_z(self, qz_x: dist.Distribution, log_pz_c: Tensor) -> Tensor:
        dkl_z = -qz_x.entropy() - torch.logsumexp(log_pz_c - math.log(self.k), dim=-1)
        return dkl_z.sum()

    def evidence_lower_bound(
        self, x: Tensor, y: torch.Tensor | None = None
    ) -> tuple[Tensor, dist.Categorical]:
        qz_x = self.qz(x, y)
        z = qz_x.rsample()

        log_pz_c = self.log_pz_c(z)

        local_ll = self.px(z, y).log_prob(x).sum()
        local_kl = self.dkl_z(qz_x, log_pz_c)
        elbo = (local_ll - local_kl) / z.shape[0]

        return elbo, self.qc(log_pz_c)


class VampPriorUnchained(VampPrior):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_L = torch.nn.Parameter(torch.zeros(self.k, self.dim_z, self.dim_z))

    def log_pz_c(self, z: Tensor) -> Tensor:
        mu = self.qz(self.u).loc
        L = (
            self.log_L.tril(-1)
            + (self.log_L.diagonal(dim1=-2, dim2=-1)).exp().diag_embed()
        )
        pz_c = dist.MultivariateNormal(loc=mu, scale_tril=L)
        return pz_c.log_prob(z[:, None])
