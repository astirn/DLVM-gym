import math
from abc import ABC, abstractmethod

import torch
import torch.distributions as dist
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal, Normal

from .abstract import AbstractVampPrior


class VampPrior(AbstractVampPrior, ABC):

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        num_data: int,
        truncation_level: int,
        u_initial: torch.Tensor,
        **kwargs,
    ):
        super().__init__(dim_x, dim_z, num_data, truncation_level, u_initial)

    @abstractmethod
    def px(self, z: torch.Tensor) -> Distribution:
        pass

    @abstractmethod
    def qz(self, x: torch.Tensor) -> Normal | MultivariateNormal:
        pass

    @property
    @abstractmethod
    def u(self) -> torch.Tensor:
        pass

    def log_pz_c(self, z: Tensor) -> Tensor:
        pz_c = self.qz(self.u)
        return pz_c.log_prob(z[:, None])

    def dkl_z(self, qz_x: dist.Distribution, log_pz_c: Tensor) -> Tensor:
        dkl_z = -qz_x.entropy() - torch.logsumexp(log_pz_c - math.log(self.k), dim=-1)
        return dkl_z.sum()

    def evidence_lower_bound(self, x: Tensor) -> tuple[Tensor, dict[str, Distribution]]:
        qz_x = self.qz(x)
        z = qz_x.rsample()

        log_pz_c = self.log_pz_c(z)

        local_ll = self.px(z).log_prob(x).sum()
        local_kl = self.dkl_z(qz_x, log_pz_c)
        elbo = (local_ll - local_kl) / z.shape[0]

        return elbo, {"qz_x": qz_x}
