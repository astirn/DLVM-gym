import torch
from torch import Tensor
from torch.distributions import (
    Beta,
    Categorical,
    Distribution,
    Gamma,
    MultivariateNormal,
    kl_divergence,
)

from .vamppriors import VampPrior, VampPriorUnchained


class DirichletProcessVampPrior(VampPrior):

    def __init__(
        self,
        *,
        num_data: int,
        concentration: float | None = None,
        concentration_prior_params: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_data = num_data

        if (concentration is None) == (concentration_prior_params is None):
            raise ValueError(
                "Specify either concentration or concentration_prior_params"
            )
        self.concentration = concentration
        if concentration_prior_params is not None:
            self.register_buffer("a", torch.as_tensor(concentration_prior_params[0]))
            self.register_buffer("b", torch.as_tensor(concentration_prior_params[1]))
            self.log_alpha = torch.nn.Parameter(torch.tensor(0.0))
            self.log_beta = torch.nn.Parameter(torch.tensor(0.0))

        self.log_gamma = torch.nn.Parameter(torch.zeros(self.k, 2))

    @property
    def gamma(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.log_gamma.exp()[:, 0], self.log_gamma.exp()[:, 1]

    def log_v_and_log_vm1(self) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_1, gamma_2 = self.gamma
        log_v = (torch.digamma(gamma_1) - torch.digamma(gamma_1 + gamma_2))[:-1]
        log_v = torch.concat([log_v, torch.as_tensor([0]).to(log_v.device)])
        log_vm1 = (torch.digamma(gamma_2) - torch.digamma(gamma_1 + gamma_2))[:-1]
        log_vm1 = torch.concat(
            [log_vm1, torch.as_tensor([-torch.inf]).to(log_vm1.device)]
        )
        return log_v, log_vm1

    @property
    def pkappa(self) -> Gamma:
        return Gamma(self.get_buffer("a"), self.get_buffer("b"))

    @property
    def qkappa(self) -> Gamma:
        return Gamma(self.log_alpha.exp(), self.log_beta.exp())

    def qc(self, log_pz_c: Tensor, log_v: Tensor, log_vm1: Tensor) -> Categorical:
        log_probs = (
            log_pz_c
            + log_v
            + torch.concat(
                [torch.zeros_like(log_vm1)[:1], torch.cumsum(log_vm1, dim=0)[:-1]]
            )
        )
        return Categorical(logits=log_probs)

    def dkl_z(self, qz_x, qc_z, log_pz_c) -> torch.Tensor:
        dkl_z = -qz_x.entropy() - (qc_z.probs * log_pz_c).sum(dim=-1)
        return dkl_z

    def dkl_c(self, qc_z: Categorical, log_v: Tensor, log_vm1: Tensor) -> torch.Tensor:
        qc_gtj_z = torch.fliplr(torch.cumsum(torch.fliplr(qc_z.probs), dim=1)[:, :-1])
        qc_gtj_z = torch.concat([qc_gtj_z, torch.zeros_like(qc_gtj_z)[:, :1]], dim=1)
        dkl_c = (
            -qc_z.entropy()
            - (qc_z.probs * log_v).sum(dim=1)
            - (qc_gtj_z[:, :-1] * log_vm1[:-1]).sum(dim=1)
        )
        return dkl_c

    def dkl_kappa(self) -> float | torch.Tensor:
        if self.concentration is None:
            return kl_divergence(self.qkappa, self.pkappa)
        else:
            return 0.0

    def dkl_v(self, log_vm1: Tensor) -> torch.Tensor:
        gamma_1, gamma_2 = self.gamma
        qv = Beta(gamma_1, gamma_2)
        if self.concentration is None:
            alpha, beta = self.log_alpha.exp(), self.log_beta.exp()
            E_log_kappa = torch.digamma(alpha) - self.log_beta
            E_kappa = alpha / beta
            return (
                -qv.entropy()[:-1].sum()
                - (self.k - 1) * E_log_kappa
                - (E_kappa - 1) * log_vm1[:-1].sum()
            )
        else:
            pv = Beta(1, self.concentration)
            return kl_divergence(qv, pv)[:-1].sum()

    def dkl(self, qz_x: Distribution) -> Tensor:
        z = qz_x.rsample()
        log_pz_c = self.log_pz_c(z)
        log_v, log_vm1 = self.log_v_and_log_vm1()
        qc_z = self.qc(log_pz_c, log_v, log_vm1)
        local_kl = self.dkl_z(qz_x, qc_z, log_pz_c) + self.dkl_c(qc_z, log_v, log_vm1)
        global_kl = self.dkl_kappa() + self.dkl_v(log_vm1)
        return local_kl.mean() + global_kl / self.num_data

    def local_distributions(self, qz_x: MultivariateNormal) -> dict[str, Distribution]:
        z = qz_x.rsample()
        log_pz_c = self.log_pz_c(z)
        log_v, log_vm1 = self.log_v_and_log_vm1()
        qc_z = self.qc(log_pz_c, log_v, log_vm1)
        return {"qc_z": qc_z}


class DirichletProcessVampPriorUnchained(DirichletProcessVampPrior, VampPriorUnchained):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_pz_c(self, z: Tensor) -> Tensor:
        return VampPriorUnchained.log_pz_c(self, z)
