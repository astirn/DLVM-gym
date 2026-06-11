import torch
import torch.distributions as dist
import torch.nn.functional as F
from torchmetrics import Metric


class ModelMetrics(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("log_marginal", default=torch.tensor(0.0))
        self.add_state("posterior_predictive", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))

    def update(
        self,
        *,
        log_marginal: torch.Tensor,
        posterior_predictive: torch.Tensor,
    ) -> None:
        self.log_marginal += log_marginal
        self.posterior_predictive += posterior_predictive
        self.num_samples += 1

    def compute(self) -> dict[str, torch.Tensor]:
        num_samples = getattr(self, "num_samples")
        return {
            "log_marginal": getattr(self, "log_marginal") / num_samples,
            "posterior_predictive": getattr(self, "posterior_predictive") / num_samples,
        }
