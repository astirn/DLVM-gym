import math

import torch
import torch.distributions as dist
from torch import Tensor

from .abstract import AbstractVampPrior


class VampPrior(AbstractVampPrior):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_pz_c(self, z: Tensor) -> Tensor:
        pz_c = self._encoder(self.u)
        return pz_c.log_prob(z[:, None])

    def qc(self, log_pz_c: Tensor) -> dist.Categorical:
        return dist.Categorical(logits=log_pz_c)

    def dkl(self, qz_x: dist.Distribution) -> Tensor:
        log_pz_c = self.log_pz_c(qz_x.rsample())
        dkl_z = -qz_x.entropy() - torch.logsumexp(log_pz_c - math.log(self.k), dim=-1)
        return dkl_z.mean()


class VampPriorUnchained(VampPrior):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_L = torch.nn.Parameter(torch.zeros(self.k, self.dim_z, self.dim_z))

    def log_pz_c(self, z: Tensor) -> Tensor:
        mu = self._encoder(self.u).loc
        L = (
            self.log_L.tril(-1)
            + (self.log_L.diagonal(dim1=-2, dim2=-1)).exp().diag_embed()
        )
        pz_c = dist.MultivariateNormal(loc=mu, scale_tril=L)
        return pz_c.log_prob(z[:, None])
