import math

import torch
import torch.distributions as dist
from torch import Tensor

from .abstract import AbstractVampPrior


class VampPrior(AbstractVampPrior):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def pz_c(self) -> dist.MultivariateNormal:
        return self.qz_u

    @property
    def pz(self) -> dist.Distribution:
        pz_c = self.pz_c
        return dist.MixtureSameFamily(
            dist.Categorical(logits=torch.zeros(self.k, device=pz_c.loc.device)),
            pz_c,
        )

    def cluster_responsibilities(self, z: Tensor) -> dist.Categorical:
        return dist.Categorical(logits=self.log_pz_c(z))

    def dkl(self, qz_x: dist.Distribution, z_sample: Tensor) -> Tensor:
        log_pz_c = self.log_pz_c(z_sample)
        dkl_z = -qz_x.entropy() - torch.logsumexp(log_pz_c - math.log(self.k), dim=-1)
        return dkl_z.mean()

    def log_pz(
        self, z: Tensor, global_samples: dict[str, Tensor] | None = None
    ) -> Tensor:
        return self.pz.log_prob(z)


class VampPriorUnleashed(VampPrior):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_L = torch.nn.Parameter(torch.zeros(self.k, self.dim_z, self.dim_z))

    @property
    def pz_c(self) -> dist.MultivariateNormal:
        mu = self.qz_u.loc
        L = (
            self.log_L.tril(-1)
            + (self.log_L.diagonal(dim1=-2, dim2=-1)).exp().diag_embed()
        )
        return dist.MultivariateNormal(loc=mu, scale_tril=L)
