from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig
from toolz import valmap

from src.data import BaseDataModule
from src.model_wrapper import ModelWrapper


def fit(cfg: DictConfig) -> tuple[BaseDataModule, ModelWrapper, Trainer]:
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    model = ModelWrapper(
        model=instantiate(cfg.model, datamodule=datamodule),
        validation_metrics=list(map(instantiate, cfg.metrics.validation_metrics)),
        test_metrics=list(map(instantiate, cfg.metrics.test_metrics)),
        optimizer_params=dict(cfg.optimizer),
        scheduler_params=(
            dict(cfg.scheduler) if cfg.get("scheduler") is not None else None
        ),
    )

    callbacks = list(valmap(instantiate, cfg.callbacks).values())
    trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(model=model, datamodule=datamodule)

    return datamodule, model, trainer
