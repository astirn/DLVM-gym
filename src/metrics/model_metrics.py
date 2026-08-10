from math import log

import torch
from toolz import valmap
from torch import Tensor
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal
from torchmetrics import Metric

from src.models.abstract import DeepLatentVariableModel


class ModelMetrics(Metric):

    log_locals: Tensor
    log_posterior_predictive: Tensor
    num_samples: Tensor
    distortion: Tensor
    z_samples: list[Tensor]
    log_qz_x: list[Tensor]
    log_pz: list[Tensor]
    qz_loc: list[Tensor]
    qz_scale_tril: list[Tensor]

    def __init__(
        self, k: int, global_batch_size: int = 128, marginal_kl_batch_size: int = 64
    ):
        super().__init__()
        self.k = k
        self.global_batch_size = global_batch_size
        self.marginal_kl_batch_size = marginal_kl_batch_size
        self.add_state("log_locals", default=torch.zeros(k))
        self.add_state("log_posterior_predictive", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))
        self.add_state("distortion", default=torch.tensor(0.0))
        self.add_state("z_samples", default=[])
        self.add_state("log_qz_x", default=[])
        self.add_state("log_pz", default=[])
        self.add_state("qz_loc", default=[])
        self.add_state("qz_scale_tril", default=[])
        self.global_samples: dict[str, Tensor] | None = None
        self.log_globals: Tensor | None = None

    def update(
        self, batch: dict[str, torch.Tensor], model: DeepLatentVariableModel
    ) -> None:
        if model.prior.has_global_latent_variables:
            if self.global_samples is None and self.log_globals is None:
                self.global_samples, self.log_globals = (
                    model.prior.global_importance_sampling(self.k)
                )

        qz = model.qz(**batch)
        z = qz.sample((self.k,))
        log_qz = qz.log_prob(z)
        log_pxz = model.px(z=z, **batch).log_prob(batch["x"])

        # for models without global latent variables, this computes the prior predictive
        # for models with global latent variables, this computes the posterior predictive
        # using analytically marginalized p(z) = E[p(z|global latents)] by not providing global samples
        log_pz = model.prior.log_pz(z)
        log_w = log_pxz + log_pz - log_qz
        self.log_posterior_predictive += (
            torch.logsumexp(log_w, dim=-2) - log(self.k)
        ).sum()

        if model.prior.has_global_latent_variables:
            # this computes the prior predictive via importance sampling over the global latents
            log_px = []
            for i in range(0, self.k, self.global_batch_size):
                j = min(i + self.global_batch_size, self.k)
                globals_batch = valmap(lambda x: x[i:j], self.global_samples)
                log_w = log_pxz + model.prior.log_pz(z, globals_batch) - log_qz
                log_px.append(torch.logsumexp(log_w, dim=-2) - log(self.k))
            self.log_locals += torch.vstack(log_px).sum(dim=-1)

        self.num_samples += batch["x"].shape[0]
        self.distortion -= log_pxz.mean(dim=0).sum()

        # accumulate samples and their log-densities for ELBO surgery (Hoffman & Johnson, 2016)
        self.z_samples.append(z[0])
        self.log_qz_x.append(log_qz[0])
        self.log_pz.append(log_pz[0])
        self.qz_loc.append(qz.loc)
        self.qz_scale_tril.append(qz.scale_tril)

    def _log_qz(self) -> Tensor:
        """log q(z) of the aggregate posterior 1/N sum_n q(z|x_n) at the stored samples."""
        z = torch.cat(self.z_samples)
        qz = MixtureSameFamily(
            Categorical(logits=torch.zeros(len(z), device=z.device)),
            MultivariateNormal(
                torch.cat(self.qz_loc), scale_tril=torch.cat(self.qz_scale_tril)
            ),
        )
        return torch.cat(
            [qz.log_prob(zs) for zs in z.split(self.marginal_kl_batch_size)]
        )

    def compute(self) -> dict[str, torch.Tensor]:
        log_posterior_predictive = self.log_posterior_predictive / self.num_samples

        # ELBO surgery (Hoffman & Johnson, 2016): rate = marginal KL + I[z;n]
        log_pz = torch.cat(self.log_pz)
        rate = (torch.cat(self.log_qz_x) - log_pz).mean()
        marginal_kl = (self._log_qz() - log_pz).mean()

        # without global latents the prior and posterior predictive densities coincide
        metrics = {
            "log_marginal": log_posterior_predictive,
            "log_posterior_predictive": log_posterior_predictive,
            "distortion": self.distortion / self.num_samples,
            "rate": rate,
            "marginal_kl": marginal_kl,
            "mutual_information": rate - marginal_kl,
        }

        # with global latents, the prior predictive uses importance sampling over the global latents
        if (log_globals := self.log_globals) is not None:
            log_w = (self.log_locals + log_globals).double()
            metrics["log_marginal"] = (
                (torch.logsumexp(log_w, dim=0) - log(self.k)) / self.num_samples
            ).float()
            metrics["global_ess"] = torch.exp(
                2 * torch.logsumexp(log_w, dim=0) - torch.logsumexp(2 * log_w, dim=0)
            ).float()

        return metrics

    def reset(self) -> None:
        super().reset()
        self.global_samples = None
        self.log_globals = None
