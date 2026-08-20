from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as dist
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from lightning import Trainer
from matplotlib.gridspec import GridSpec
from omegaconf import DictConfig
from toolz import interleave, merge_with, pipe
from toolz.curried import partition
from torch.utils.data import TensorDataset

from src.data import GaussianMixtureDataModule
from src.distributions import concat_distributions
from src.fit import fit
from src.model_wrapper import ModelWrapper

COLOR_PALETTE = pipe(
    sns.color_palette("tab20c", n_colors=20) + sns.color_palette("tab20b", n_colors=20),
    partition(4),
    interleave,
    list,
)


class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, hidden_dim: int, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.mu = nn.Linear(hidden_dim, dim_z)
        self.log_sigma = nn.Linear(hidden_dim, dim_z)

    def forward(self, x: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = self.log_sigma(out).exp()
        return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, hidden_dim: int, **kwargs):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(dim_z, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, dim_x),
        )
        self.log_sigma = nn.Parameter(torch.zeros(dim_x))

    def forward(self, z: torch.Tensor) -> dist.Distribution:
        sigma = self.log_sigma.exp()
        return dist.MultivariateNormal(self.mu(z), scale_tril=torch.diag_embed(sigma))


def plot_results(
    datamodule: GaussianMixtureDataModule, model_wrapper: ModelWrapper, trainer: Trainer
):
    prior_name = model_wrapper.model.prior.__class__.__name__
    fig_path = Path(HydraConfig.get().runtime.output_dir) / f"toy_plot_{prior_name}.pdf"

    dataset = datamodule.data["predict"]
    if not isinstance(dataset, TensorDataset):
        raise ValueError("Dataset is not a TensorDataset")
    x = dataset.tensors[0].numpy(force=True)

    distributions = trainer.predict(model_wrapper, datamodule.predict_dataloader())
    distributions = merge_with(concat_distributions, distributions)

    z = distributions["qz"].sample()
    z_min, z_max = min(z).item(), max(z).item()
    z_sweep = torch.linspace(z_min, z_max, 1000).unsqueeze(-1).unsqueeze(-1)
    z_plot = z_sweep.squeeze(-1).squeeze(-1).numpy(force=True)

    cluster_responsibilities = distributions["cluster_responsibilities"].probs
    cluster_assignments = cluster_responsibilities.argmax(axis=1).numpy(force=True)
    cluster_usage = cluster_responsibilities.mean(axis=0).numpy(force=True)

    pz = model_wrapper.model.prior.pz
    if not isinstance(pz, dist.MixtureSameFamily):
        raise ValueError("Prior is not a MixtureSameFamily distribution")
    cluster_proportions = pz.mixture_distribution.probs.numpy(force=True)
    pz = pz.component_distribution.log_prob(z_sweep).exp().numpy(force=True)

    fig = plt.figure(figsize=(6, 10))
    gs = GridSpec(4, 2, figure=fig)

    ax = fig.add_subplot(gs[:2, :])
    ax.set_title("Data colored by cluster assignments")
    ax.set_xlabel("$x_1$")
    ax.set_xticks([])
    ax.set_ylabel("$x_2$")
    ax.set_yticks([])
    sns.scatterplot(
        x=x[:, 0],
        y=x[:, 1],
        hue=cluster_assignments,
        hue_order=range(len(COLOR_PALETTE)),
        palette=COLOR_PALETTE,
        ax=ax,
        legend=False,
    )

    ax = fig.add_subplot(gs[2, :])
    ax.set_title("Cluster components weighted by prior")
    for i, line_data in enumerate((cluster_proportions * pz).T):
        sns.lineplot(
            x=z_plot,
            y=line_data,
            ax=ax,
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            legend=False,
        )
    ax.set_xlabel("$z$")

    ax = fig.add_subplot(gs[3, :])
    ax.set_title("Cluster components weighted by usage")
    for i, line_data in enumerate((cluster_usage * pz).T):
        sns.lineplot(
            x=z_plot,
            y=line_data,
            ax=ax,
            color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            legend=False,
        )
    ax.set_xlabel("$z$")

    plt.tight_layout()
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)


@hydra.main(version_base=None, config_path="../../conf", config_name="experiments/toy")
def main(cfg: DictConfig) -> None:
    datamodule, model_wrapper, trainer = fit(cfg)
    if isinstance(datamodule, GaussianMixtureDataModule):
        plot_results(datamodule, model_wrapper, trainer)
        if HydraConfig.get().mode == RunMode.RUN:
            plt.show()
    else:
        raise ValueError("Dataset is not a GaussianMixtureDataModule")


if __name__ == "__main__":
    main()
