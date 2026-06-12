import torch
from torch.distributions import MultivariateNormal, kl_divergence

from .abstracts import Prior


class StandardNormalPrior(Prior):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.pz = MultivariateNormal(loc=torch.zeros(dim), scale_tril=torch.eye(dim))

    def dkl(self, qz_x: MultivariateNormal) -> torch.Tensor:
        return kl_divergence(qz_x, self.pz).mean()
