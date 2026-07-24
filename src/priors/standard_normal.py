import torch
import torch.distributions as dist
from torch import Tensor

from .abstract import Prior


class StandardNormalPrior(Prior):
    def __init__(self, *, dim_z: int, **kwargs):
        super().__init__()
        self.register_buffer("loc", torch.zeros(dim_z))
        self.register_buffer("scale_tril", torch.eye(dim_z))

    @property
    def pz(self) -> dist.Distribution:
        return dist.MultivariateNormal(
            loc=self.get_buffer("loc"), scale_tril=self.get_buffer("scale_tril")
        )

    def component_distribution(self, z: Tensor) -> dist.Categorical:
        return dist.Categorical(probs=torch.ones_like(z[:, :1]))

    def dkl(self, qz_x: dist.Distribution, z_sample: Tensor) -> Tensor:
        return dist.kl_divergence(qz_x, self.pz).mean()

    def log_pz(self, z: Tensor, globals: dict[str, Tensor] | None = None) -> Tensor:
        return self.pz.log_prob(z)
