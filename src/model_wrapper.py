from abc import ABC
from typing import Any

from hydra.utils import instantiate
from lightning import LightningModule
from torch import Tensor
from torchmetrics import Metric

from src.models.abstract import DeepLatentVariableModel


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
        self.validation_metrics = ModuleList(validation_metrics or [])
        self.test_metrics = ModuleList(test_metrics or [])
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        loss = self.model.loss(batch)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.loss(batch)
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["x"].shape[0],
        )
        for metric in self.validation_metrics:
            if isinstance(metric, Metric):
                metric.update(batch, self.model)

    def on_validation_epoch_end(self) -> None:
        for metric in self.validation_metrics:
            if isinstance(metric, Metric):
                self.log_dict(metric.compute(), on_epoch=True, prog_bar=True)
                metric.reset()

    def test_step(self, batch, batch_idx):
        for metric in self.test_metrics:
            if isinstance(metric, Metric):
                metric.update(batch, self.model)

    def on_test_epoch_end(self) -> None:
        for metric in self.test_metrics:
            if isinstance(metric, Metric):
                self.log_dict(metric.compute(), on_epoch=True)
                metric.reset()

    def predict_step(self, batch, batch_idx) -> dict[str, Any]:
        qz = self.model.qz(**batch)
        z = qz.sample()
        px = self.model.px(z=z, **batch)
        predictions = {
            "qz": qz,
            "px": px,
        }

        try:
            predictions["component_distribution"] = (
                self.model.prior.component_distribution(z=z)
            )
        except NotImplementedError:
            pass

        return predictions

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
