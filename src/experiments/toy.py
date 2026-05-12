from pathlib import Path

import hydra
import lightning
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as dist
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import Trainer
from matplotlib.gridspec import GridSpec
from omegaconf import DictConfig
from toolz import merge_with, valmap
from torch.utils.data import TensorDataset

from constants import COLOR_PALETTE
from src.data import GaussianMixtureDataModule
from src.model_wrapper import ModelWrapper


class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x, hidden_dim),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden_dim, dim_z)
        self.log_sigma = nn.Linear(hidden_dim, dim_z)

    def forward(self, x: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = self.log_sigma(out).exp()
        return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, hidden_dim: int):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(dim_z, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim_x),
        )
        self.log_sigma = nn.Parameter(torch.as_tensor(0.0))

    def forward(self, z: torch.Tensor) -> dist.Distribution:
        scale = self.log_sigma.exp()
        return dist.Independent(dist.Normal(self.mu(z), scale), 1)


def plot_results(
    datamodule: GaussianMixtureDataModule, model: ModelWrapper, trainer: Trainer
):
    fig_path = Path(HydraConfig.get().runtime.output_dir) / "toy_plot.pdf"

    dataset = datamodule.data["test"]
    if not isinstance(dataset, TensorDataset):
        raise ValueError("Dataset is not a TensorDataset")
    x = dataset.tensors[0].numpy(force=True)
    c = dataset.tensors[1].numpy(force=True)

    predictions = trainer.predict(model, datamodule.predict_dataloader())
    predictions = merge_with(lambda x: torch.vstack(x).cpu().numpy(), predictions)
    predicted_cluster_assignments = predictions["cluster_probs"].argmax(axis=1)

    fig = plt.figure(figsize=(6, 10))
    gs = GridSpec(4, 2, figure=fig)

    ax = fig.add_subplot(gs[:2, :])
    ax.set_title("True Cluster Assignments")
    ax.set_xlabel("$x_1$")
    ax.set_xticks([])
    ax.set_ylabel("$x_2$")
    ax.set_yticks([])
    sns.scatterplot(
        x=x[:, 0],
        y=x[:, 1],
        hue=c,
        hue_order=range(datamodule.num_clusters),
        palette=COLOR_PALETTE,
        ax=ax,
        legend=False,
    )

    ax = fig.add_subplot(gs[2:, :])
    ax.set_title("Predicted Cluster Assignments")
    ax.set_xlabel("$x_1$")
    ax.set_xticks([])
    ax.set_ylabel("$x_2$")
    ax.set_yticks([])
    sns.scatterplot(
        x=x[:, 0],
        y=x[:, 1],
        hue=predicted_cluster_assignments,
        hue_order=range(model.model.k),
        palette=COLOR_PALETTE,
        ax=ax,
        legend=False,
    )

    plt.tight_layout()

    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)

    plt.show()


def train(cfg: DictConfig) -> tuple[GaussianMixtureDataModule, ModelWrapper, Trainer]:
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.data)
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


@hydra.main(version_base=None, config_path="../../conf", config_name="experiment/toy")
def main(cfg: DictConfig) -> None:
    datamodule, model, trainer = train(cfg)
    plot_results(datamodule, model, trainer)


if __name__ == "__main__":
    main()
