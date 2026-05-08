import torch
import torch.distributions as dist
from torch import Tensor

from .vamprior import VampPrior


class DirichletProcessVampPrior(VampPrior):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.log_alpha_v = torch.nn.Parameter(torch.zeros(self.k))
        self.log_beta_v = torch.nn.Parameter(torch.ones(self.k))

    def log_pz_c(self, z: Tensor) -> Tensor:
        pz_c = self.qz(self.u)
        return pz_c.log_prob(z[:, None])

    def log_v_and_log_vm1(self) -> tuple[torch.Tensor, torch.Tensor]:
        alpha, beta = torch.exp(self.log_alpha_v), torch.exp(self.log_beta_v)
        log_v = (torch.digamma(alpha) - torch.digamma(alpha + beta))[:-1]
        log_v = torch.concat([log_v, torch.as_tensor([0]).to(log_v.device)])
        log_vm1 = (torch.digamma(beta) - torch.digamma(alpha + beta))[:-1]
        log_vm1 = torch.concat(
            [log_vm1, torch.as_tensor([-torch.inf]).to(log_vm1.device)]
        )
        return log_v, log_vm1

    def qc(self, log_pz_c: Tensor) -> dist.Categorical:
        log_v, log_vm1 = self.log_v_and_log_vm1()
        log_probs = (
            log_pz_c
            + log_v
            + torch.concat(
                [torch.zeros_like(log_vm1)[:1], torch.cumsum(log_vm1, dim=0)[:-1]]
            )
        )
        return dist.Categorical(logits=log_probs)

    def dkl_c(self, qc_z: dist.Categorical):
        log_v, log_vm1 = self.log_v_and_log_vm1()
        qc_gtj_z = torch.fliplr(torch.cumsum(torch.fliplr(qc_z.probs), dim=1)[:, :-1])
        qc_gtj_z = torch.concat([qc_gtj_z, torch.zeros_like(qc_gtj_z)[:, :1]], dim=1)
        dkl_c = (
            -qc_z.entropy()
            - (qc_z.probs * log_v).sum(dim=1)
            - (qc_gtj_z[:, :-1] * log_vm1[:-1]).sum(dim=1)
        )
        return dkl_c.sum()

    def dkl_alpha0(self):
        if False:  # self.concentration_prior:
            raise NotImplementedError("Concentration prior not implemented")
            # a_q, b_q = self.a_and_b_q_alpha0()
            # a_p = self.get_buffer("a_p")
            # b_p = self.get_buffer("b_p")
            # return (
            #     a_p * (b_q / b_p).log()
            #     + torch.lgamma(a_p)
            #     - torch.lgamma(a_q)
            #     - (a_p - a_q) * torch.digamma(a_q)
            #     + a_q * (b_p / b_q - 1)
            # )
        else:
            return 0

    def dkl_v(self):
        q = dist.Beta(torch.exp(self.log_alpha_v), torch.exp(self.log_beta_v))
        if False:
            raise NotImplementedError("Concentration prior not implemented")
            # a_q, b_q = self.a_and_b_q_alpha0()
            # E_log_alpha0 = b_q.log() - torch.digamma(a_q)
            # E_alpha0 = b_q / (a_q - 1)
            # _, log_vm1 = self.log_v_and_log_vm1()
            # return (
            #     -q.entropy()[:-1].sum()
            #     - (self.k - 1) * E_log_alpha0
            #     - (E_alpha0 - 1) * log_vm1[:-1].sum()
            # )
        else:
            p = dist.Beta(1, 0.1)
            return dist.kl_divergence(q, p)[:-1].sum()

    def evidence_lower_bound(self, x: Tensor, c: torch.Tensor | None = None) -> Tensor:
        qz_x = self.qz(x, c)
        z = qz_x.rsample()

        log_pz_c = self.log_pz_c(z)
        qc_z = self.qc(log_pz_c)

        local_ll = self.px(z, c).log_prob(x).sum()
        local_kl = self.dkl_z(qz_x, log_pz_c) + self.dkl_c(qc_z)
        global_kl = self.dkl_alpha0() + self.dkl_v()
        elbo = (local_ll - local_kl) / z.shape[0] - global_kl / self.num_data

        return elbo
