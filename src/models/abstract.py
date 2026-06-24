from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module

from src.priors import Prior


class DeepLatentVariableModel(Module, ABC):

    prior: Prior

    @abstractmethod
    def loss(self, batch: dict[str, Tensor]) -> Tensor: ...
