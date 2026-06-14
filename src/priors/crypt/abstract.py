from abc import ABC
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
        **kwargs,
    ):
        super().__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.k = k
        self._encoder = encoder
        self._u = torch.nn.Parameter(torch.randn(k, dim_x))
        self._u_constraint = u_constraint

    @property
    def u(self) -> Tensor:
        return self._u_constraint(self._u)
