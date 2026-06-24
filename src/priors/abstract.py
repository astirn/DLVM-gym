from abc import ABC, abstractmethod

from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Module


class Prior(Module, ABC):

    @property
    def pz(self) -> Distribution:
        raise NotImplementedError  # optionally required for some metrics

    def component_scores(self, z: Tensor) -> Tensor:
        raise NotImplementedError  # optionally required for some metrics

    @abstractmethod  # required for model training
    def dkl(self, qz_x: Distribution) -> Tensor: ...
