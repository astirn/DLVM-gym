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

    def evidence_lower_bound(self, x: Tensor, c: torch.Tensor | None = None) -> Tensor:
        qz_x = self.qz(x, c)
        z = qz_x.rsample()

        log_pz_c = self.log_pz_c(z)

        local_ll = self.px(z, c).log_prob(x).sum()
        local_kl = self.dkl_z(qz_x, log_pz_c)
        elbo = (local_ll - local_kl) / z.shape[0]

        return elbo
