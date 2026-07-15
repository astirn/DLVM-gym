from abc import ABC, abstractmethod

from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal
from torch.nn import Module

from src.priors import Prior


class DeepLatentVariableModel(Module, ABC):

    prior: Prior

    def __init__(
        self,
        kl_warmup_epochs: int | None = None,
        kl_warmup_steps: int | None = None,
        **kwargs,
    ):
        super().__init__()
        if kl_warmup_epochs is not None and kl_warmup_steps is not None:
            raise ValueError("Use either kl_warmup_epochs or kl_warmup_steps, not both")
        elif kl_warmup_epochs is not None and kl_warmup_epochs < 0:
            raise ValueError("kl_warmup_epochs must be non-negative")
        elif kl_warmup_steps is not None and kl_warmup_steps < 0:
            raise ValueError("kl_warmup_steps must be non-negative")
        else:
            self.kl_warmup_steps = kl_warmup_steps
            self.kl_warmup_epochs = kl_warmup_epochs

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
