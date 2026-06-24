import torch
from torch.distributions import Distribution, MultivariateNormal, kl_divergence

from .abstract import Prior


class StandardNormalPrior(Prior):
    def __init__(self, *, dim_z: int, **kwargs):
        super().__init__()
        self.register_buffer("loc", torch.zeros(dim_z))
        self.register_buffer("scale_tril", torch.eye(dim_z))

    @property
    def pz(self) -> Distribution:
        return MultivariateNormal(
            loc=self.get_buffer("loc"), scale_tril=self.get_buffer("scale_tril")
        )

    def dkl(self, qz_x: MultivariateNormal) -> torch.Tensor:
        return kl_divergence(qz_x, self.pz).mean()
