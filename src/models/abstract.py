from abc import ABC, abstractmethod

from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Module

from src.priors import Prior


class DeepLatentVariableModel(Module, ABC):

    prior: Prior

    def global_distributions(self) -> dict[str, Distribution]:
        return self.prior.global_distributions()

    @abstractmethod
    def local_distributions(
        self, batch: dict[str, Tensor]
    ) -> dict[str, Distribution]: ...

    @abstractmethod
    def loss(self, batch: dict[str, Tensor]) -> Tensor: ...
