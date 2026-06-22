from abc import ABC, abstractmethod

from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from torch.nn import Module


class Prior(Module, ABC):

    @abstractmethod
    def dkl(self, qz_x: Distribution) -> Tensor: ...

    @abstractmethod
    def global_distributions(self) -> dict[str, Distribution]: ...

    @abstractmethod
    def local_distributions(
        self, qz_x: MultivariateNormal
    ) -> dict[str, Distribution]: ...
