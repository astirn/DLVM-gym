import hydra
import lightning
import torch
import torch.distributions as dist
import torch.nn as nn
from hydra.utils import instantiate
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from toolz import valmap

from src.data.abstract import AbstractDataModule
from src.model_wrapper import ModelWrapper


class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x, hidden_dim),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_dim, dim_z)
        self.sigma_uc = nn.Linear(hidden_dim, dim_z)

    def forward(self, x: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = self.sigma_uc(out).exp()
        return dist.MultivariateNormal(mu, covariance_matrix=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, hidden_dim: int):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(dim_z, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim_x),
        )
        self.sigma_uc = nn.Parameter(torch.as_tensor(0.0))

    def forward(self, z: torch.Tensor) -> dist.Distribution:
        scale = self.sigma_uc.exp().sqrt()
        return dist.Independent(dist.Normal(self.mu(z), scale), 1)


def initial_u_tensor(batch_x: torch.Tensor, truncation_level: int) -> torch.Tensor:
    n = min(truncation_level, batch_x.shape[0])
    return batch_x[:n].clone()


def train(cfg: DictConfig) -> tuple[LightningDataModule, LightningModule]:
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    datamodule: AbstractDataModule = instantiate(cfg.data)
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
    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(model=model, datamodule=datamodule)

    # if cfg.get("analysis"):
    #     log_dir = Path(HydraConfig.get().runtime.output_dir)
    #     instantiate(cfg.analysis)(log_dir=log_dir)

    return datamodule, model


@hydra.main(version_base=None, config_path="../../conf", config_name="experiment/toy")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
