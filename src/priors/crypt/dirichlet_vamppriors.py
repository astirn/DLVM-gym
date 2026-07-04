import torch
from torch import Tensor
from torch.distributions import (
    Categorical,
    Dirichlet,
    Distribution,
    MixtureSameFamily,
    kl_divergence,
)

from .vamppriors import AbstractVampPrior, VampPriorUnchained


class DirichletVampPrior(AbstractVampPrior):

    def __init__(self, *, concentration: float, num_data: int, **kwargs):
        super().__init__(**kwargs)
        self.concentration = concentration
        self.num_data = num_data
        self._log_alpha = torch.nn.Parameter(torch.zeros(self.k))

    @property
    def alpha(self) -> Tensor:
        return self._log_alpha.exp()

    @property
    def palpha(self) -> Dirichlet:
        return Dirichlet(
            concentration=self.concentration * torch.ones_like(self.alpha) / self.k
        )

    @property
    def qalpha(self) -> Dirichlet:
        return Dirichlet(concentration=self.alpha)

    def e_log_pi(self) -> Tensor:
        alpha = self.alpha
        return torch.digamma(alpha) - torch.digamma(alpha.sum())

    def qc(self, log_pz_c: Tensor, e_log_pi: Tensor) -> Categorical:
        log_probs = log_pz_c + e_log_pi
        return Categorical(logits=log_probs)

    def dkl_pi(self) -> Tensor:
        return kl_divergence(self.qalpha, self.palpha)

    def dkl_c(self, qc_z: Categorical, e_log_pi: Tensor) -> Tensor:
        return -qc_z.entropy() - (qc_z.probs * e_log_pi).sum(dim=-1)

    def dkl_z(self, qz_x: Distribution, qc_z: Categorical, log_pz_c: Tensor) -> Tensor:
        return -qz_x.entropy() - (qc_z.probs * log_pz_c).sum(dim=-1)

    def dkl(self, qz_x: Distribution, z_sample: Tensor) -> Tensor:
        log_pz_c = self.log_pz_c(z_sample)
        e_log_pi = self.e_log_pi()
        qc_z = self.qc(log_pz_c, e_log_pi)
        local_kl = self.dkl_z(qz_x, qc_z, log_pz_c) + self.dkl_c(qc_z, e_log_pi)
        global_kl = self.dkl_pi()
        return local_kl.mean() + global_kl / self.num_data

    def component_distribution(self, z: Tensor) -> Categorical:
        log_pz_c = self.log_pz_c(z)
        e_log_pi = self.e_log_pi()
        return self.qc(log_pz_c, e_log_pi)

    @property
    def pz(self) -> Distribution:
        probs = self.alpha / self.alpha.sum()
        return MixtureSameFamily(Categorical(probs=probs), self.pz_c)


class DirichletVampPriorUnchained(DirichletVampPrior, VampPriorUnchained):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_pz_c(self, z: Tensor) -> Tensor:
        return VampPriorUnchained.log_pz_c(self, z)
