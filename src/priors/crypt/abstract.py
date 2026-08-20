from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from toolz import identity
from torch import Tensor
from torch.distributions import MultivariateNormal

from src.priors.abstract import Prior

CONSTRAINT_FN = Callable[[Tensor], Tensor]


class AbstractVampPrior(Prior, ABC):

    def __init__(
        self,
        *,
        dim_z: int,
        k: int,
        encoder: torch.nn.Module,
        u_initial: Tensor | list[Tensor],
        u_constraint: CONSTRAINT_FN | list[CONSTRAINT_FN] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.dim_z = dim_z
        self.k = k
        self._encoder = encoder
        if isinstance(u_initial, Tensor):
            u_initial = [u_initial]
        self._u = torch.nn.ParameterList([torch.nn.Parameter(u) for u in u_initial])
        if u_constraint is None:
            u_constraint = [identity] * len(self._u)
        elif not isinstance(u_constraint, list):
            u_constraint = [u_constraint]
        self._u_constraint = u_constraint
        if len(self._u_constraint) != len(self._u):
            raise ValueError(
                "u_constraint must be a list of the same length as u_initial"
            )

    @property
    def u(self) -> list[Tensor]:
        return [fn(u) for fn, u in zip(self._u_constraint, self._u)]

    @property
    def qu(self) -> MultivariateNormal:
        was_training = self._encoder.training
        self._encoder.eval()
        qu = self._encoder(*self.u)
        if was_training:
            self._encoder.train()
        return qu

    @property
    @abstractmethod
    def pz_c(self) -> MultivariateNormal: ...

    def log_pz_c(self, z: Tensor) -> Tensor:
        return self.pz_c.log_prob(z[:, None])
