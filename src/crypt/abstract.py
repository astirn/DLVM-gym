from abc import ABC, abstractmethod

import torch
import torch.distributions as dist
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal, Normal


class AbstractVampPrior(torch.nn.Module, ABC):

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        num_data: int,
        truncation_level: int,
        u_initial: torch.Tensor,
        **kwargs,
    ):
        super().__init__()

        self.dim_z = dim_z
        self.num_data = num_data
        self.k = truncation_level
        self.u_uc = nn.Parameter(u_initial)

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

    @abstractmethod
    def log_pz_c(self, z: Tensor) -> Tensor:
        pass

    @abstractmethod
    def dkl_z(self, qz_x: dist.Distribution, log_pz: Tensor, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def evidence_lower_bound(self, x: Tensor) -> tuple[Tensor, dict[str, Distribution]]:
        pass
