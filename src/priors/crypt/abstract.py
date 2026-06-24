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
        dim_x: int,
        dim_z: int,
        k: int,
        encoder: Callable[[Tensor], MultivariateNormal],
        u_constraint: Callable[[Tensor], Tensor] = identity,
        u_initial: Tensor | None = None,
        **kwargs,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.k = k
        self._encoder = encoder
        if u_initial is not None:
            if u_initial.shape == (k, dim_x):
                self._u = torch.nn.Parameter(u_initial)
            else:
                raise ValueError(
                    f"u_initial must have shape (k, dim_x), but got {u_initial.shape}"
                )
        else:
            self._u = torch.nn.Parameter(torch.randn(k, dim_x))
        self._u_constraint = u_constraint

    @property
    @abstractmethod
    def pz_c(self) -> MultivariateNormal: ...

    @property
    def u(self) -> Tensor:
        return self._u_constraint(self._u)

    def log_pz_c(self, z: Tensor) -> Tensor:
        return self.pz_c.log_prob(z[:, None])
