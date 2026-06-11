from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.distributions as dist
from hydra.utils import instantiate
from toolz import identity
from torch import Tensor


class AbstractVampPrior(torch.nn.Module, ABC):

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        decoder_params: dict[str, Any],
        encoder_params: dict[str, Any],
        k: int,
        num_data: int,
        u_constraint: Callable[[Tensor], Tensor] = identity,
        **kwargs,
    ):
        super().__init__()
        self._decoder = instantiate(decoder_params, dim_x=dim_x, dim_z=dim_z)
        self._encoder = instantiate(encoder_params, dim_x=dim_x, dim_z=dim_z)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.num_data = num_data
        self.k = k
        self._u = torch.nn.Parameter(torch.randn(k, dim_x))
        self._u_constraint = u_constraint

    @abstractmethod
    def evidence_lower_bound(
        self, x: Tensor, y: Tensor | None = None
    ) -> tuple[Tensor, dist.Categorical]: ...

    def loss_and_cluster_probs(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        elbo, qc_z = self.evidence_lower_bound(batch["x"], batch.get("y"))
        return -elbo, qc_z.probs

    def px(self, z: Tensor, y: Tensor | None = None) -> dist.Distribution:
        zy = torch.concat([z, y], dim=1) if y is not None else z
        return self._decoder(zy)

    @abstractmethod
    def qc(self, *args) -> dist.Categorical: ...

    def qz(self, x: Tensor, y: Tensor | None = None) -> dist.MultivariateNormal:
        xy = torch.concat([x, y], dim=1) if y is not None else x
        return self._encoder(xy)

    @property
    def u(self) -> Tensor:
        return self._u_constraint(self._u)
