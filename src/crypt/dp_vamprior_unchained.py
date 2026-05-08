import torch
import torch.distributions as dist
from torch import Tensor

from .dp_vamprior import DirichletProcessVampPrior
from .vamprior_unchained import VampPriorUnchained


class DirichletProcessVampPriorUnchained(DirichletProcessVampPrior, VampPriorUnchained):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_pz_c(self, z: Tensor) -> Tensor:
        return VampPriorUnchained.log_pz_c(self, z)
