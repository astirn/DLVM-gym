import math
from typing import Any

import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch import Tensor

from src.utils import softplus_inverse

from .abstract import AbstractVampPrior


class VampPriorMixtureModel(AbstractVampPrior):
    """VMM prior with optional empirical-Bayes alternating inference."""

    def __init__(
        self,
        *,
        concentration: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.concentration = concentration

        self.pi_logits = torch.nn.Parameter(torch.zeros(self.k))
        self._precision_params = torch.nn.Parameter(
            torch.zeros(self.k, self.dim_z, self.dim_z)
            + softplus_inverse(torch.ones(self.k, self.dim_z)).diag_embed()
        )

        self.register_buffer(
            "wishart_scale_tril",
            torch.eye(self.dim_z)
            * (self.k ** (1 / self.dim_z))
            / ((self.dim_z + 2) ** 0.5),
        )

    @property
    def alternating_inference(self) -> bool:
        return True

    def empirical_bayes_params(self) -> list[Tensor]:
        return [
            p
            for name, p in self.named_parameters()
            if p.requires_grad and not name.startswith("_encoder.")
        ]

    @property
    def precision_cholesky(self) -> Tensor:
        # matches https://github.com/tensorflow/probability/blob/v0.23.0/tensorflow_probability/python/bijectors/fill_scale_tril.py
        diagonal = F.softplus(self._precision_params.diagonal(dim1=-2, dim2=-1)) + 1e-5
        return diagonal.diag_embed() + self._precision_params.tril(-1)

    def log_pz_c(self, z: Tensor) -> Tensor:
        precision = self.precision_cholesky @ self.precision_cholesky.mT
        d = z.unsqueeze(1) - self.qz_u.loc.unsqueeze(0)
        return 0.5 * (
            torch.logdet(precision / 2 / math.pi).unsqueeze(0)
            - (d.unsqueeze(-2) @ precision @ d.unsqueeze(-1)).squeeze(-1).squeeze(-1)
            - torch.diagonal(precision @ self.qz_u.covariance_matrix, dim1=-2, dim2=-1)
            .sum(-1)
            .unsqueeze(0)
        )

    def log_prior_prob_pi(self) -> Tensor:
        alpha = torch.full_like(self.pi_logits, self.concentration / self.k)
        pi = F.softmax(self.pi_logits, dim=-1)
        return dist.Dirichlet(alpha).log_prob(pi)

    def log_prior_prob_mu(self) -> Tensor:
        constant = self.dim_z * math.log(2.0 * math.pi)
        squared_norm = (self.qz_u.loc**2).sum(dim=-1)
        trace_cov = torch.diagonal(self.qz_u.covariance_matrix, dim1=-2, dim2=-1).sum(
            -1
        )
        return -0.5 * (constant + (squared_norm + trace_cov)).sum()

    def log_prior_prob_precision(self) -> Tensor:
        prior_precision = dist.Wishart(
            df=self.dim_z + 2,
            scale_tril=self.get_buffer("wishart_scale_tril"),
        )
        precision = self.precision_cholesky @ self.precision_cholesky.mT
        return prior_precision.log_prob(precision).sum()

    def empirical_bayes_loss(self, batch: dict[str, Any]) -> Tensor:
        with torch.no_grad():
            z = self._encoder(**batch).sample()

        log_prob_z_c = self.log_pz_c(z)
        log_prob_c = F.log_softmax(self.pi_logits, dim=-1)
        qc = F.softmax(log_prob_z_c + log_prob_c, dim=-1)

        local_loss = -(qc * (log_prob_z_c + log_prob_c)).sum(dim=-1).mean()
        global_loss = (
            self.log_prior_prob_pi()
            + self.log_prior_prob_mu()
            + self.log_prior_prob_precision()
        )
        loss = local_loss - global_loss / z.shape[0]

        return loss

    @property
    def pz_c(self) -> dist.MultivariateNormal:
        cov = (
            torch.cholesky_inverse(self.precision_cholesky)
            + self.qz_u.covariance_matrix
        )
        scale_tril = torch.linalg.cholesky(cov)
        return dist.MultivariateNormal(loc=self.qz_u.loc, scale_tril=scale_tril)

    @property
    def pz(self) -> dist.Distribution:
        return dist.MixtureSameFamily(
            dist.Categorical(logits=self.pi_logits), self.pz_c
        )

    def dkl(self, qz_x: dist.Distribution, z_sample: Tensor) -> Tensor:
        return (-qz_x.entropy() - self.pz.log_prob(z_sample)).mean()

    def cluster_responsibilities(self, z: Tensor) -> dist.Categorical:
        return dist.Categorical(
            logits=self.log_pz_c(z) + F.log_softmax(self.pi_logits, dim=-1)
        )

    def log_pz(
        self, z: Tensor, global_samples: dict[str, Tensor] | None = None
    ) -> Tensor:
        return self.pz.log_prob(z)
