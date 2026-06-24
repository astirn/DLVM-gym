from abc import ABC, abstractmethod

from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Module


class Prior(Module, ABC):

    @property
    @abstractmethod
    def pz(self) -> Distribution: ...

    @abstractmethod
    def dkl(self, qz_x: Distribution) -> Tensor: ...
