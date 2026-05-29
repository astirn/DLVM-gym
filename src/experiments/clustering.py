import hydra
import lightning
import torch
import torch.distributions as dist
import torch.nn as nn
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import DictConfig
from toolz import valmap

from src.data import AbstractDataModule
from src.model_wrapper import ModelWrapper


class Encoder(nn.Module):
    def __init__(
        self,
        dim_x: int,
        dim_z: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x, 500),
            nn.ELU(),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Linear(500, 2000),
            nn.ELU(),
        )
        self.mu = nn.Linear(2000, dim_z)
        self.log_sigma = nn.Linear(2000, dim_z)

    def forward(self, x: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = self.log_sigma(out).exp()
        return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(dim_z, 2000),
            nn.ELU(),
            nn.Linear(2000, 500),
            nn.ELU(),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Linear(500, dim_x),
        )
        self.log_sigma = nn.Parameter(torch.as_tensor(0.0))

    def forward(self, z: torch.Tensor) -> dist.Distribution:
        scale = self.log_sigma.exp()
        return dist.Independent(dist.Normal(self.mu(z), scale), 1)


def train(cfg: DictConfig) -> tuple[AbstractDataModule, ModelWrapper, Trainer]:
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    train_iter = iter(datamodule.train_dataloader())
    u = torch.tensor([])
    while len(u) < int(cfg.truncation_level):
        batch = next(train_iter)
        u = torch.cat([u, batch["x"]])
    u = u[: int(cfg.truncation_level)]

    model = ModelWrapper(
        model=instantiate(cfg.model, u=u),
        optimizer_params=dict(cfg.optimizer),
        scheduler_params=(
            dict(cfg.scheduler) if cfg.get("scheduler") is not None else None
        ),
    )

    callbacks = list(valmap(instantiate, cfg.callbacks).values())
    trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(model=model, datamodule=datamodule)

    return datamodule, model, trainer


@hydra.main(
    version_base=None, config_path="../../conf", config_name="experiment/clustering"
)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
