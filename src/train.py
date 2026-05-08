import logging
from pathlib import Path

import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from toolz import valmap

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> tuple[LightningDataModule, LightningModule]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = instantiate(cfg.data)
    model: LightningModule = instantiate(cfg.model)
    callbacks = list(valmap(instantiate, cfg.callbacks).values())
    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(model=model, datamodule=datamodule)

    if cfg.get("analysis"):
        log_dir = Path(HydraConfig.get().runtime.output_dir)
        instantiate(cfg.analysis)(log_dir=log_dir)

    return datamodule, model


@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
