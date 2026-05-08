from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.distributions as dist
from toolz import identity
from torch import Tensor


class AbstractVampPrior(torch.nn.Module, ABC):

    def __init__(
        self,
        decoder: torch.nn.Module,
        encoder: torch.nn.Module,
        num_data: int,
        u: Tensor,
        u_constraint: Callable[[Tensor], Tensor] = identity,
        **kwargs,
    ):
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.num_data = num_data
        self.k = u.shape[0]
        self._u = torch.nn.Parameter(u)
        self._u_constraint = u_constraint

    def px(self, z: Tensor, c: Tensor | None = None) -> dist.Distribution:
        zc = torch.concat([z, c], dim=1) if c is not None else z
        return self.decoder(zc)

    def qz(self, x: Tensor, c: Tensor | None = None) -> dist.MultivariateNormal:
        xc = torch.concat([x, c], dim=1) if c is not None else x
        return self.encoder(xc)

    @property
    def u(self) -> Tensor:
        return self._u_constraint(self._u)

    @abstractmethod
    def qc(self, log_pz_c: Tensor) -> dist.Categorical:
        pass

    @abstractmethod
    def evidence_lower_bound(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        pass
