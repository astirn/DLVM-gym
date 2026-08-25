from abc import ABC
from typing import Any

import torch
from hydra.utils import instantiate
from lightning import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerPLType, OptimizerLRScheduler
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
        self.automatic_optimization = False

        self.model = model
        self._optimizer_params = optimizer_params
        self._scheduler_params = scheduler_params
        self._validation_metrics = ModuleList(validation_metrics or [])
        self._test_metrics = ModuleList(test_metrics or [])
        self._test_prefix = "test"

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

    @property
    def model_params(self) -> list[Tensor]:
        if self.model.prior.alternating_inference:
            return [
                p
                for name, p in self.model.named_parameters()
                if p.requires_grad and not name.startswith("prior.")
            ]

        return [p for p in self.model.parameters() if p.requires_grad]

    @property
    def prior_params(self) -> list[Tensor]:
        return self.model.prior.empirical_bayes_params()

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        optimizers = self.optimizers()
        optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
        params = self.model_params

        loss = self.model.loss(
            batch, epoch=self.trainer.current_epoch, global_step=self.global_step
        )
        grads = torch.autograd.grad(loss, params, allow_unused=True)

        optimizer.zero_grad(set_to_none=True)
        for p, g in zip(params, grads):
            if g is not None:
                p.grad = g
        optimizer.step()

        self.log(
            "train/loss", loss.detach(), on_epoch=True, on_step=False, prog_bar=True
        )

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.model.prior.alternating_inference:

            optimizers = self.optimizers()
            if not isinstance(optimizers, list):
                raise ValueError(
                    "Optimizers must be a list when alternating inference."
                )
            optimizer = optimizers[1]
            params = self.prior_params

            loss = self.model.prior.empirical_bayes_loss(batch)
            grads = torch.autograd.grad(loss, params, allow_unused=True)

            optimizer.zero_grad(set_to_none=True)
            for p, g in zip(params, grads):
                if g is not None:
                    p.grad = g
            optimizer.step()

            self.log(
                "train/prior_loss",
                loss.detach(),
                on_epoch=True,
                on_step=False,
                prog_bar=False,
            )

        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        [sch.step() for sch in schedulers if isinstance(sch, LRSchedulerPLType)]

    @property
    def validation_metrics(self) -> ModuleList:
        return self._validation_metrics

    @validation_metrics.setter
    def validation_metrics(self, value: Metric | list[Metric]) -> None:
        self._set_metrics("_validation_metrics", value)

    def validation_step(self, batch, batch_idx):
        loss = self.model.loss(
            batch, epoch=self.trainer.current_epoch, global_step=self.global_step
        )
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

    @property
    def test_prefix(self) -> str:
        return self._test_prefix

    @test_prefix.setter
    def test_prefix(self, value: str) -> None:
        self._test_prefix = value

    def test_step(self, batch, batch_idx):
        for metric in self.test_metrics:
            if isinstance(metric, Metric):
                metric.update(batch, self.model)

    def on_test_epoch_end(self) -> None:
        for metric in self.test_metrics:
            if isinstance(metric, Metric):
                metric_dict = keymap(
                    lambda x: f"{self.test_prefix}/{x}", metric.compute()
                )
                self.log_dict(metric_dict, on_epoch=True)
                metric.reset()

    def predict_step(self, batch, batch_idx) -> dict[str, Any]:
        qz = self.model.qz(**batch)
        z = qz.sample()
        px = self.model.px(z=z, **batch)
        predictions = {"qz": qz, "px": px}

        try:
            predictions["cluster_responsibilities"] = (
                self.model.prior.cluster_responsibilities(z=z)
            )
        except NotImplementedError:
            pass

        return predictions

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.model.prior.alternating_inference:
            optimizer1 = instantiate(self._optimizer_params, params=self.model_params)
            optimizer2 = instantiate(self._optimizer_params, params=self.prior_params)

            if self._scheduler_params is not None:
                scheduler1 = instantiate(self._scheduler_params, optimizer=optimizer1)
                scheduler2 = instantiate(self._scheduler_params, optimizer=optimizer2)

                return {
                    "optimizer": optimizer1,
                    "lr_scheduler": {
                        "scheduler": scheduler1,
                        "interval": "step",
                    },
                }, {
                    "optimizer": optimizer2,
                    "lr_scheduler": {
                        "scheduler": scheduler2,
                        "interval": "step",
                    },
                }

            return optimizer1, optimizer2

        optimizer = instantiate(self._optimizer_params, params=self.model_params)

        if self._scheduler_params is not None:
            scheduler = instantiate(self._scheduler_params, optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return optimizer
