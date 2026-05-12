import torch
import torch.distributions as dist
from torch import Tensor

from .vamp_priors import VampPrior, VampPriorUnchained


class DirichletProcessVampPrior(VampPrior):

    def __init__(
        self,
        *,
        concentration: float | None = None,
        concentration_prior_params: tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if (concentration is None) == (concentration_prior_params is None):
            raise ValueError(
                "Specify either concentration or concentration_prior_params"
            )

        self.concentration = concentration
        if concentration_prior_params is not None:
            self.register_buffer("a", torch.as_tensor(concentration_prior_params[0]))
            self.register_buffer("b", torch.as_tensor(concentration_prior_params[1]))
            self.log_alpha_minus_1 = torch.nn.Parameter(torch.tensor(0.0))
            self.log_beta = torch.nn.Parameter(torch.tensor(0.0))

        self.log_gamma = torch.nn.Parameter(torch.zeros(self.k, 2))

    def log_v_and_log_vm1(self) -> tuple[torch.Tensor, torch.Tensor]:
        gamma_1, gamma_2 = self.log_gamma[:, 0].exp(), self.log_gamma[:, 1].exp()
        log_v = (torch.digamma(gamma_1) - torch.digamma(gamma_1 + gamma_2))[:-1]
        log_v = torch.concat([log_v, torch.as_tensor([0]).to(log_v.device)])
        log_vm1 = (torch.digamma(gamma_2) - torch.digamma(gamma_1 + gamma_2))[:-1]
        log_vm1 = torch.concat(
            [log_vm1, torch.as_tensor([-torch.inf]).to(log_vm1.device)]
        )
        return log_v, log_vm1

    @property
    def pkappa(self) -> dist.InverseGamma:
        return dist.InverseGamma(self.get_buffer("a"), self.get_buffer("b"))

    @property
    def qkappa(self) -> dist.InverseGamma:
        return dist.InverseGamma(1 + self.log_alpha_minus_1.exp(), self.log_beta.exp())

    def qc(self, log_pz_c: Tensor, log_v: Tensor, log_vm1: Tensor) -> dist.Categorical:
        log_probs = (
            log_pz_c
            + log_v
            + torch.concat(
                [torch.zeros_like(log_vm1)[:1], torch.cumsum(log_vm1, dim=0)[:-1]]
            )
        )
        return dist.Categorical(logits=log_probs)

    def dkl_z(self, qz_x, qc_z, log_pz_c) -> torch.Tensor:
        dkl_z = -qz_x.entropy() - (qc_z.probs * log_pz_c).sum(dim=-1)
        return dkl_z.sum()

    def dkl_c(
        self, qc_z: dist.Categorical, log_v: Tensor, log_vm1: Tensor
    ) -> torch.Tensor:
        qc_gtj_z = torch.fliplr(torch.cumsum(torch.fliplr(qc_z.probs), dim=1)[:, :-1])
        qc_gtj_z = torch.concat([qc_gtj_z, torch.zeros_like(qc_gtj_z)[:, :1]], dim=1)
        dkl_c = (
            -qc_z.entropy()
            - (qc_z.probs * log_v).sum(dim=1)
            - (qc_gtj_z[:, :-1] * log_vm1[:-1]).sum(dim=1)
        )
        return dkl_c.sum()

    def dkl_kappa(self) -> float | torch.Tensor:
        if self.concentration is None:
            return dist.kl_divergence(self.qkappa, self.pkappa)
        else:
            return 0.0

    def dkl_v(self, log_vm1: Tensor) -> torch.Tensor:
        gamma_1, gamma_2 = self.log_gamma[:, 0].exp(), self.log_gamma[:, 1].exp()
        qv = dist.Beta(gamma_1, gamma_2)
        if self.concentration is None:
            alpha, beta = 1 + self.log_alpha_minus_1.exp(), self.log_beta.exp()
            E_log_kappa = self.log_beta - torch.digamma(alpha)
            E_kappa = beta / (alpha - 1)
            return (
                -qv.entropy()[:-1].sum()
                - (self.k - 1) * E_log_kappa
                - (E_kappa - 1) * log_vm1[:-1].sum()
            )
        else:
            pv = dist.Beta(1, self.concentration)
            return dist.kl_divergence(qv, pv)[:-1].sum()

    def evidence_lower_bound(self, x: Tensor, c: torch.Tensor | None = None) -> Tensor:
        qz_x = self.qz(x, c)
        z = qz_x.rsample()

        log_pz_c = self.log_pz_c(z)
        log_v, log_vm1 = self.log_v_and_log_vm1()
        qc_z = self.qc(log_pz_c, log_v, log_vm1)

        local_ll = self.px(z, c).log_prob(x).sum()
        local_kl = self.dkl_z(qz_x, qc_z, log_pz_c) + self.dkl_c(qc_z, log_v, log_vm1)
        global_kl = self.dkl_kappa() + self.dkl_v(log_vm1)
        elbo = (local_ll - local_kl) / z.shape[0] - global_kl / self.num_data

        return elbo


class DirichletProcessVampPriorUnchained(DirichletProcessVampPrior, VampPriorUnchained):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_pz_c(self, z: Tensor) -> Tensor:
        return VampPriorUnchained.log_pz_c(self, z)
