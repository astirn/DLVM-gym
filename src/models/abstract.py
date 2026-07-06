from abc import ABC, abstractmethod

from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from torch.nn import Module

from src.priors import Prior


class DeepLatentVariableModel(Module, ABC):

    prior: Prior

    @abstractmethod
    def qz(self, **kwargs) -> MultivariateNormal: ...

    @abstractmethod
    def px(self, **kwargs) -> Distribution: ...

    @abstractmethod
    def loss(self, batch: dict[str, Tensor], global_step: int) -> Tensor: ...
