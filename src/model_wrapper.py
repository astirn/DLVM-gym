from abc import ABC
from typing import Any

from hydra.utils import instantiate
from lightning import LightningModule
from toolz import keymap
from torch import Tensor
from torch.nn import ModuleList
from torchmetrics import Metric

from src.models.abstract import DeepLatentVariableModel


class ModelWrapper(LightningModule, ABC):
    def __init__(
        self,
        model: DeepLatentVariableModel,
        *,
        optimizer_params: dict[str, Any] | None = None,
        scheduler_params: dict[str, Any] | None = None,
        validation_metrics: list[Metric] | None = None,
        test_metrics: list[Metric] | None = None,
        **kwargs,
    ) -> None:
        LightningModule.__init__(self)
        self.model = model
        self._optimizer_params = optimizer_params
        self._scheduler_params = scheduler_params
        self._validation_metrics = ModuleList(validation_metrics or [])
        self._test_metrics = ModuleList(test_metrics or [])

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "validation_metrics":
            self._set_metrics("_validation_metrics", value)
            return
        if name == "test_metrics":
            self._set_metrics("_test_metrics", value)
            return
        super().__setattr__(name, value)

    def _set_metrics(self, attr: str, value: Metric | list[Metric]) -> None:
        metrics = [value] if isinstance(value, Metric) else value
        super().__setattr__(attr, ModuleList(metrics))

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        loss = self.model.loss(batch)
        self.log("train/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    @property
    def validation_metrics(self) -> ModuleList:
        return self._validation_metrics

    @validation_metrics.setter
    def validation_metrics(self, value: Metric | list[Metric]) -> None:
        self._set_metrics("_validation_metrics", value)

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
                metric_dict = keymap(lambda x: f"val/{x}", metric.compute())
                self.log_dict(metric_dict, on_epoch=True, prog_bar=True)
                metric.reset()

    @property
    def test_metrics(self) -> ModuleList:
        return self._test_metrics

    @test_metrics.setter
    def test_metrics(self, value: Metric | list[Metric]) -> None:
        self._set_metrics("_test_metrics", value)

    def test_step(self, batch, batch_idx):
        for metric in self.test_metrics:
            if isinstance(metric, Metric):
                metric.update(batch, self.model)

    def on_test_epoch_end(self) -> None:
        for metric in self.test_metrics:
            if isinstance(metric, Metric):
                metric_dict = keymap(lambda x: f"test/{x}", metric.compute())
                self.log_dict(metric_dict, on_epoch=True)
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

        optimizer = instantiate(self._optimizer_params, params=self.parameters())

        if self._scheduler_params is None:
            return optimizer

        scheduler = instantiate(self._scheduler_params, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
