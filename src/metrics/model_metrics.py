from math import log

import torch
from torchmetrics import Metric

from src.models.abstract import DeepLatentVariableModel


class ModelMetrics(Metric):
    def __init__(self, k_local: int, k_global: int = 1_000):
        super().__init__()
        self.k_local = k_local
        self.k_global = k_global
        self.add_state("log_locals", default=torch.zeros(k_global))
        self.add_state("num_samples", default=torch.tensor(0.0))
        self.global_samples: dict[str, torch.Tensor] | None = None
        self.log_globals: torch.Tensor | None = None

    def update(
        self, batch: dict[str, torch.Tensor], model: DeepLatentVariableModel
    ) -> None:
        if model.prior.has_global_latent_variables:
            if self.global_samples is None and self.log_globals is None:
                self.global_samples, self.log_globals = (
                    model.prior.global_importance_sampling(self.k_global)
                )

        qz = model.qz(**batch)
        z = qz.sample((self.k_local,))
        log_pz = model.prior.log_pz(
            z, self.global_samples
        )  # [n,k,b] if globals else [k,b]
        log_qz = qz.log_prob(z)  # always[k,b]
        log_pxz = model.px(z=z, **batch).log_prob(batch["x"])  # always [k,b]

        log_w = log_pxz + log_pz - log_qz
        log_px = torch.logsumexp(log_w, dim=-2) - log(self.k_local)

        if log_px.ndim == 2:  # model has global latent variables
            self.log_locals += log_px.sum(dim=-1)
        else:  # model lacks global latent variables
            self.log_locals[0] += log_px.sum()

        self.num_samples += batch["x"].shape[0]

    def compute(self) -> dict[str, torch.Tensor]:
        if self.log_globals is not None:
            log_px = torch.logsumexp(self.log_locals + self.log_globals, dim=0) - log(
                self.k_global
            )
        else:
            log_px = self.log_locals[0]
        return {"log_marginal": log_px / self.num_samples}

    def reset(self) -> None:
        super().reset()
        self.global_samples = None
        self.log_globals = None
