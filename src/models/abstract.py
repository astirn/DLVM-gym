from abc import ABC, abstractmethod

from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from torch.nn import Module

from src.priors import Prior


class DeepLatentVariableModel(Module, ABC):

    prior: Prior
    kl_warmup_epochs: int | None
    kl_warmup_steps: int | None

    @abstractmethod
    def qz(self, **kwargs) -> MultivariateNormal: ...

    @abstractmethod
    def px(self, **kwargs) -> Distribution: ...

    @abstractmethod
    def loss(
        self, batch: dict[str, Tensor], *, epoch: int, global_step: int
    ) -> Tensor: ...

    def beta(self, epoch: int, global_step: int) -> float:
        if self.kl_warmup_epochs is not None and self.kl_warmup_epochs > 0:
            return min(1.0, epoch / self.kl_warmup_epochs)
        elif self.kl_warmup_steps is not None and self.kl_warmup_steps > 0:
            return min(1.0, global_step / self.kl_warmup_steps)
        else:
            return 1.0
