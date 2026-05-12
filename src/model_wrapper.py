from abc import ABC
from typing import Any, Protocol

from hydra.utils import instantiate
from lightning import LightningModule
from torch import Tensor
from torch.distributions import Categorical

from src.metrics import ClusteringMetrics, ModelMetrics


class ModelProtocol(Protocol):
    def loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Categorical]: ...

    @property
    def k(self) -> int: ...


class ModelWrapper(LightningModule, ABC):
    def __init__(
        self,
        model: ModelProtocol,
        *,
        optimizer_params: dict[str, Any] | None = None,
        scheduler_params: dict[str, Any] | None = None,
    ) -> None:
        LightningModule.__init__(self)
        self.model = model
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.clustering_metrics = ClusteringMetrics(k=self.model.k)
        self.model_metrics = ModelMetrics()

    def training_step(self, batch, batch_idx):
        loss, _ = self.model.loss(batch)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, cluster_proportions = self.model.loss(batch)
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["x"].shape[0],
        )
        self.clustering_metrics.update(
            labels=batch["c"], cluster_proportions=cluster_proportions
        )

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.clustering_metrics.compute(), on_epoch=True, prog_bar=True)
        self.clustering_metrics.reset()

    # def test_step(self, batch, batch_idx):
    #     x = batch["x"]
    #     S = self.num_sampling_steps
    #     elbo_draws = [self.vamp.evidence_lower_bound(x, None) for _ in range(S)]
    #     log_marginal = (
    #         torch.logsumexp(torch.stack(elbo_draws), dim=0)
    #         - torch.tensor(S, dtype=torch.float32, device=x.device).log()
    #     )
    #     log_marginal = log_marginal.mean()

    #     qz_x = self.vamp.qz(x, None)
    #     log_liks = [
    #         self.vamp.px(qz_x.rsample(), None).log_prob(x).sum() for _ in range(S)
    #     ]
    #     posterior_predictive = (
    #         torch.logsumexp(torch.stack(log_liks), dim=0)
    #         - torch.tensor(S, dtype=torch.float32, device=x.device).log()
    #     )
    #     posterior_predictive = posterior_predictive.mean()

    #     self.model_metrics.update(
    #         log_marginal=log_marginal, posterior_predictive=posterior_predictive
    #     )

    # def on_test_epoch_end(self) -> None:
    #     self.log_dict(self.model_metrics.compute(), on_epoch=True, prog_bar=True)
    #     self.model_metrics.reset()

    # def predict_step(self, batch, batch_idx):
    #     embeddings = self.vamp.qz(batch["x"], None).mean
    #     qc_z = self._qc_for_z_point(embeddings)
    #     return dict(embeddings=embeddings, cluster_probabilities=qc_z.probs)

    def configure_optimizers(self):

        optimizer = instantiate(self.optimizer_params, params=self.parameters())

        if self.scheduler_params is None:
            return optimizer

        scheduler = instantiate(self.scheduler_params, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
