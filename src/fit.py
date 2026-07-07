from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
from toolz import valmap

from src.data import BaseDataModule
from src.model_wrapper import ModelWrapper


def fit(cfg: DictConfig) -> tuple[BaseDataModule, ModelWrapper, Trainer]:
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    metrics = valmap(instantiate, cfg.get("metrics", {}))
    model = ModelWrapper(
        model=instantiate(cfg.model, datamodule=datamodule),
        optimizer_params=dict(cfg.optimizer),
        scheduler_params=(
            dict(cfg.scheduler) if cfg.get("scheduler") is not None else None
        ),
        **metrics,
    )

    callbacks = list(valmap(instantiate, cfg.get("callbacks", {})).values())
    loggers = list(valmap(instantiate, cfg.get("loggers", {})).values())
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    hyperparams = OmegaConf.to_container(cfg, resolve=True)
    for logger in trainer.loggers:
        logger.log_hyperparams(hyperparams)

    trainer.fit(model=model, datamodule=datamodule)

    return datamodule, model, trainer
