from abc import ABC
from typing import Any

from hydra.utils import instantiate
from lightning import LightningModule
from torch import Tensor
from torch.distributions import Distribution
from torchmetrics import Metric

from .abstract import DeepLatentVariableModel


class ModelWrapper(LightningModule, ABC):
    def __init__(
        self,
        model: DeepLatentVariableModel,
        *,
        validation_metrics: list[Metric] | None = None,
        test_metrics: list[Metric] | None = None,
        optimizer_params: dict[str, Any] | None = None,
        scheduler_params: dict[str, Any] | None = None,
    ) -> None:
        LightningModule.__init__(self)
        self.model = model
        self.validation_metrics = validation_metrics or []
        self.test_metrics = test_metrics or []
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        loss = self.model.loss(batch)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.loss(batch)
        local_distributions = self.model.local_distributions(batch)
        for metric in self.validation_metrics:
            metric.update(batch, local_distributions)
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["x"].shape[0],
        )

    def on_validation_epoch_end(self) -> None:
        for metric in self.validation_metrics:
            self.log_dict(metric.compute(), on_epoch=True, prog_bar=True)
            metric.reset()

    def predict_step(self, batch, batch_idx) -> dict[str, Distribution]:
        return self.model.local_distributions(batch)

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
