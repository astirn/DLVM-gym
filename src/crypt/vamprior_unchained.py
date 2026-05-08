import math
from abc import ABC, abstractmethod

import torch
import torch.distributions as dist
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal, Normal

from .vamprior import VampPrior


class VampPriorUnchained(VampPrior, ABC):

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
        qz_x = self.qz(self.u)
        if isinstance(qz_x, MultivariateNormal):
            L = torch.linalg.cholesky(self.qz(self.u).precision_matrix)
        elif isinstance(qz_x, Normal):
            L = qz_x.scale.diag_embed()
        else:
            raise ValueError(f"Unsupported distribution type: {type(qz_x)}")
        self.log_L = nn.Parameter(
            L.tril(-1) + torch.log(L.diagonal(dim1=-2, dim2=-1)).diag_embed(),
            requires_grad=True,
        )

    @abstractmethod
    def px(self, z: torch.Tensor) -> Distribution:
        pass

    @abstractmethod
    def qz(self, x: torch.Tensor) -> MultivariateNormal:
        pass

    @property
    @abstractmethod
    def u(self) -> torch.Tensor:
        pass

    def log_pz_c(self, z: Tensor) -> Tensor:
        mu = self.qz(self.u).loc
        L = (
            self.log_L.tril(-1)
            + (self.log_L.diagonal(dim1=-2, dim2=-1)).exp().diag_embed()
        )
        pz_c = dist.MultivariateNormal(loc=mu, scale_tril=L)
        return pz_c.log_prob(z[:, None])
