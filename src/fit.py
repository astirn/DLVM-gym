from hydra.utils import instantiate
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf, open_dict
from toolz import valmap

from src.data import BaseDataModule
from src.model_wrapper import ModelWrapper


def fit(cfg: DictConfig) -> tuple[BaseDataModule, ModelWrapper, Trainer]:
    if cfg.get("seed") is not None:
        seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    metrics = valmap(instantiate, cfg.get("metrics", {}))
    if cfg.model.prior_params.get("num_data", "???") == "???":
        with open_dict(cfg):
            cfg.model.prior_params["num_data"] = datamodule.num_data
    model_wrapper = ModelWrapper(
        model=instantiate(cfg.model, datamodule=datamodule),
        optimizer_params=dict(cfg.optimizer),
        scheduler_params=(
            dict(cfg.scheduler) if cfg.get("scheduler") is not None else None
        ),
        **dict(cfg.get("model_wrapper", {})),
        **metrics,
    )

    callbacks = list(valmap(instantiate, cfg.get("callbacks", {})).values())
    loggers = list(valmap(instantiate, cfg.get("loggers", {})).values())
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    hyperparams = OmegaConf.to_container(cfg, resolve=True)
    for logger in trainer.loggers:
        logger.log_hyperparams(hyperparams)

    trainer.fit(model=model_wrapper, datamodule=datamodule)

    return datamodule, model_wrapper, trainer
