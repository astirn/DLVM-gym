from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from toolz import identity, valmap
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
        q: Callable[..., MultivariateNormal],
        u_initial: dict[str, Tensor],
        u_constraint: dict[str, CONSTRAINT_FN] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.dim_z = dim_z
        self.k = k
        self.q = q

        self._encoder = encoder
        self._u = torch.nn.ParameterDict(valmap(torch.nn.Parameter, u_initial))
        if u_constraint is None:
            u_constraint = {k: identity for k in self._u}
        self._u_constraint = u_constraint
        if set(self._u.keys()) != set(self._u_constraint.keys()):
            raise ValueError("u_initial and u_constraint must have the same keys")

    @property
    def u(self) -> dict[str, Tensor]:
        return {k: self._u_constraint[k](self._u[k]) for k in self._u}

    @property
    def qz_u(self) -> MultivariateNormal:
        was_training = self._encoder.training
        self._encoder.eval()
        qz_u = self._encoder(**self.u)
        if was_training:
            self._encoder.train()
        return qz_u

    @property
    @abstractmethod
    def pz_c(self) -> MultivariateNormal: ...

    def log_pz_c(self, z: Tensor) -> Tensor:
        return self.pz_c.log_prob(z[:, None])
