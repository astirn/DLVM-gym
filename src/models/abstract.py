from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class DeepLatentVariableModel(Module, ABC):
    @abstractmethod
    def loss(self, batch: dict[str, Tensor]) -> Tensor: ...
