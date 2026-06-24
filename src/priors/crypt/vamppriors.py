import math

import torch
import torch.distributions as dist

from .abstract import AbstractVampPrior


class VampPrior(AbstractVampPrior):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def pz(self) -> dist.Distribution:
        return dist.MixtureSameFamily(
            dist.Categorical(
                logits=torch.zeros(self.k).to(self.u.device, non_blocking=True)
            ),
            self._encoder(self.u),
        )

    def qc(self, z: torch.Tensor) -> dist.Categorical:
        return dist.Categorical(logits=self.log_pz_c(z))

    def dkl(self, qz_x: dist.Distribution) -> torch.Tensor:
        log_pz_c = self.log_pz_c(qz_x.rsample())
        dkl_z = -qz_x.entropy() - torch.logsumexp(log_pz_c - math.log(self.k), dim=-1)
        return dkl_z.mean()


class VampPriorUnchained(VampPrior):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_L = torch.nn.Parameter(torch.zeros(self.k, self.dim_z, self.dim_z))

    def log_pz_c(self, z: torch.Tensor) -> torch.Tensor:
        mu = self._encoder(self.u).loc
        L = (
            self.log_L.tril(-1)
            + (self.log_L.diagonal(dim1=-2, dim2=-1)).exp().diag_embed()
        )
        pz_c = dist.MultivariateNormal(loc=mu, scale_tril=L)
        return pz_c.log_prob(z[:, None])
