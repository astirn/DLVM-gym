from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.distributions as dist


class Prior(torch.nn.Module, ABC):

    @abstractmethod
    def dkl(self, qz_x: dist.Distribution) -> torch.Tensor: ...


class ClusteringPrior(Prior, ABC):

    @abstractmethod
    def qc(self, log_pz_c: torch.Tensor) -> dist.Categorical: ...


PriorTypes = Union[Prior, ClusteringPrior]
