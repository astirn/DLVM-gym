import torch
import torch.distributions as dist
import torch.nn.functional as F


class NegativeBinomial(dist.Distribution):
    def __init__(self, mu: torch.Tensor, theta: torch.Tensor, eps: float = 1e-8):
        super().__init__(validate_args=False)
        self.mu = mu
        self.theta = theta
        self.eps = eps

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_p = (self.mu + self.eps).log() - (self.mu + self.theta + self.eps).log()
        log_1_minus_p = (self.theta + self.eps).log() - (
            self.mu + self.theta + self.eps
        ).log()
        log_prob = (
            +x * log_p
            + self.theta * log_1_minus_p
            + (x + self.theta).lgamma()
            - self.theta.lgamma()
            - (x + 1).lgamma()
        )
        return log_prob.sum(dim=1)


class ZeroInflatedNegativeBinomial(dist.Distribution):
    def __init__(
        self,
        mu: torch.Tensor,
        theta: torch.Tensor,
        rho: torch.Tensor,
        eps: float = 1e-8,
    ):
        super().__init__(validate_args=False)
        self.mu = mu
        self.theta = theta
        self.rho = rho
        self.eps = eps

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_p = (self.mu + self.eps).log() - (self.mu + self.theta + self.eps).log()
        log_1_minus_p = (self.theta + self.eps).log() - (
            self.mu + self.theta + self.eps
        ).log()
        zero_ll = F.softplus(-self.rho + self.theta * log_1_minus_p) - F.softplus(
            -self.rho
        )
        non_zero_ll = (
            -self.rho
            - F.softplus(-self.rho)
            + self.theta * log_1_minus_p
            + x * log_p
            + (x + self.theta).lgamma()
            - self.theta.lgamma()
            - (x + 1).lgamma()
        )
        log_prob = torch.where(torch.le(x, 1e-8), zero_ll, non_zero_ll)
        return log_prob.sum(dim=1)
