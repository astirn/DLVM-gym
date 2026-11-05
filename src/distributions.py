import math

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch.distributions.wishart import Wishart as TorchWishart


class NegativeBinomial(dist.Distribution):
    def __init__(self, mu: torch.Tensor, theta: torch.Tensor, eps: float = 1e-8):
        super().__init__(validate_args=False)
        self.mu = mu
        self.theta = theta
        self.eps = eps

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        log_p = (self.mu + self.eps).log() - (self.mu + self.theta + self.eps).log()
        log_1_minus_p = (self.theta + self.eps).log() - (
            self.mu + self.theta + self.eps
        ).log()
        log_prob = (
            +value * log_p
            + self.theta * log_1_minus_p
            + (value + self.theta).lgamma()
            - self.theta.lgamma()
            - (value + 1).lgamma()
        )
        return log_prob.sum(dim=-1)


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

    @property
    def mean(self) -> torch.Tensor:
        return (1 - torch.sigmoid(self.rho)) * self.mu

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
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
            + value * log_p
            + (value + self.theta).lgamma()
            - self.theta.lgamma()
            - (value + 1).lgamma()
        )
        log_prob = torch.where(torch.le(value, 1e-8), zero_ll, non_zero_ll)
        return log_prob.sum(dim=-1)


class Wishart(TorchWishart):
    """Wishart whose log_prob avoids slogdet, which faults on some GPUs."""

    def log_prob(self, value, value_tril=None):
        """`value_tril` is the lower Cholesky factor of `value`; pass it when known."""
        if value_tril is None:
            value_tril = torch.linalg.cholesky(value)
        if self._validate_args:
            self._validate_sample(value)
        nu = self.df  # has shape (batch_shape)
        p = self._event_shape[-1]  # has singleton shape
        scale_tril = self._unbroadcasted_scale_tril
        log_det_value = 2 * value_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        # tr(S^{-1} X) = ||L_S^{-1} L_X||_F^2
        trace_term = (
            torch.linalg.solve_triangular(scale_tril, value_tril, upper=False)
            .square()
            .sum(dim=(-2, -1))
        )
        return (
            -nu
            * (
                p * math.log(2) / 2
                + scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            )
            - torch.mvlgamma(nu / 2, p=p)
            + (nu - p - 1) / 2 * log_det_value
            - trace_term / 2
        )


def concat_distributions(distributions: list[dist.Distribution]) -> dist.Distribution:
    if not distributions:
        raise ValueError("Cannot concatenate empty list of distributions")

    first = distributions[0]

    if isinstance(first, dist.Categorical):
        categoricals: list[dist.Categorical] = [
            d for d in distributions if isinstance(d, dist.Categorical)
        ]
        if len(categoricals) != len(distributions):
            raise ValueError("All distributions must be Categorical")
        return dist.Categorical(logits=torch.concat([d.logits for d in categoricals]))

    if isinstance(first, dist.Normal):
        normals: list[dist.Normal] = [
            d for d in distributions if isinstance(d, dist.Normal)
        ]
        if len(normals) != len(distributions):
            raise ValueError("All distributions must be Normal")
        return dist.Normal(
            loc=torch.concat([d.loc for d in normals]),
            scale=torch.concat([d.scale for d in normals]),
        )

    if isinstance(first, dist.MultivariateNormal):
        multivariate_normals: list[dist.MultivariateNormal] = [
            d for d in distributions if isinstance(d, dist.MultivariateNormal)
        ]
        if len(multivariate_normals) != len(distributions):
            raise ValueError("All distributions must be MultivariateNormal")
        return dist.MultivariateNormal(
            loc=torch.concat([d.loc for d in multivariate_normals]),
            covariance_matrix=torch.concat(
                [d.covariance_matrix for d in multivariate_normals]
            ),
        )

    if isinstance(first, NegativeBinomial):
        negative_binomials: list[NegativeBinomial] = [
            d for d in distributions if isinstance(d, NegativeBinomial)
        ]
        if len(negative_binomials) != len(distributions):
            raise ValueError("All distributions must be NegativeBinomial")
        return NegativeBinomial(
            mu=torch.concat([d.mu for d in negative_binomials]),
            theta=torch.concat([d.theta for d in negative_binomials]),
        )

    if isinstance(first, ZeroInflatedNegativeBinomial):
        zero_inflated_negative_binomials: list[ZeroInflatedNegativeBinomial] = [
            d for d in distributions if isinstance(d, ZeroInflatedNegativeBinomial)
        ]
        if len(zero_inflated_negative_binomials) != len(distributions):
            raise ValueError("All distributions must be ZeroInflatedNegativeBinomial")
        return ZeroInflatedNegativeBinomial(
            mu=torch.concat([d.mu for d in zero_inflated_negative_binomials]),
            theta=torch.concat([d.theta for d in zero_inflated_negative_binomials]),
            rho=torch.concat([d.rho for d in zero_inflated_negative_binomials]),
        )

    raise ValueError(f"Unsupported distribution type: {type(first)}")
