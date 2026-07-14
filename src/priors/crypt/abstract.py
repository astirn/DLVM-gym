from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from toolz import identity
from torch import Tensor
from torch.distributions import MultivariateNormal

from src.priors.abstract import Prior


class AbstractVampPrior(Prior, ABC):

    def __init__(
        self,
        *,
        dim_z: int,
        k: int,
        encoder: Callable[[Tensor], MultivariateNormal],
        u_initial: Tensor,
        u_constraint: Callable[[Tensor], Tensor] = identity,
        **kwargs,
    ):
        super().__init__()
        self.dim_z = dim_z
        self.k = k
        self._encoder = encoder
        self._u = torch.nn.Parameter(u_initial)
        self._u_constraint = u_constraint

    @property
    def u(self) -> Tensor:
        return self._u_constraint(self._u)

    @property
    def qu(self) -> MultivariateNormal:
        was_training = self._encoder.training
        self._encoder.eval()
        qu = self._encoder(self.u)
        if was_training:
            self._encoder.train()
        return qu

    @property
    @abstractmethod
    def pz_c(self) -> MultivariateNormal: ...

    def log_pz_c(self, z: Tensor) -> Tensor:
        return self.pz_c.log_prob(z[:, None])
