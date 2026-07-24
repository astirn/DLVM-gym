from math import log

import torch
from toolz import valmap
from torchmetrics import Metric

from src.models.abstract import DeepLatentVariableModel


class ModelMetrics(Metric):
    def __init__(self, k: int, global_batch_size: int = 128):
        super().__init__()
        self.k = k
        self.global_batch_size = global_batch_size
        self.add_state("log_locals", default=torch.zeros(k))
        self.add_state("num_samples", default=torch.tensor(0.0))
        self.global_samples: dict[str, torch.Tensor] | None = None
        self.log_globals: torch.Tensor | None = None

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

        if not model.prior.has_global_latent_variables:
            log_pz = model.prior.log_pz(z)
            log_w = log_pxz + log_pz - log_qz
            log_px = torch.logsumexp(log_w, dim=-2) - log(self.k)
            self.log_locals[0] += log_px.sum()

        else:
            log_px = []
            for i in range(0, self.k, self.global_batch_size):
                j = min(i + self.global_batch_size, self.k)
                globals_batch = valmap(lambda x: x[i:j], self.global_samples)
                log_pz = model.prior.log_pz(z, globals_batch)
                log_w = log_pxz + log_pz - log_qz
                log_px.append(torch.logsumexp(log_w, dim=-2) - log(self.k))
            log_px = torch.vstack(log_px)
            self.log_locals += log_px.sum(dim=-1)

        self.num_samples += batch["x"].shape[0]

    def compute(self) -> dict[str, torch.Tensor]:
        if self.log_globals is not None:
            log_px = torch.logsumexp(self.log_locals + self.log_globals, dim=0) - log(
                self.k
            )
        else:
            log_px = self.log_locals[0]
        return {"log_marginal": log_px / self.num_samples}

    def reset(self) -> None:
        super().reset()
        self.global_samples = None
        self.log_globals = None
