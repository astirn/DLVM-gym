from math import log

import torch
from torchmetrics import Metric

from src.models.abstract import DeepLatentVariableModel


class ModelMetrics(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("log_marginal", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))

    def update(
        self, batch: dict[str, torch.Tensor], model: DeepLatentVariableModel
    ) -> None:
        k = 1_000
        qz = model.qz(**batch)
        z = qz.rsample((k,))
        log_pz = model.prior.pz.log_prob(z)
        log_qz = qz.log_prob(z)
        log_pxz = model.px(z=z, **batch).log_prob(batch["x"])
        log_w = log_pxz + log_pz - log_qz
        log_px = (torch.logsumexp(log_w, dim=0) - log(k)).sum()
        self.log_marginal += log_px
        self.num_samples += batch["x"].shape[0]

    def compute(self) -> dict[str, torch.Tensor]:
        num_samples = getattr(self, "num_samples")
        return {
            "log_marginal": getattr(self, "log_marginal") / num_samples,
        }
