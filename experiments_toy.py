import argparse
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
import torch.distributions as dist
import torch.nn as nn

from constants import COLOR_PALETTE, TEST_CASES
from cytoolz import merge_with
from matplotlib.gridspec import GridSpec
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from utils import get_trainer
from models import VampPrior

CLUSTER_STD = 0.5
HIDDEN_DIM = 100
LEARNING_RATE = 5e-4
PATIENCE = 1_000
TRUNCATION_LEVEL = 40

class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x, HIDDEN_DIM),
            nn.Tanh(),
        )
        self.mu = nn.Linear(HIDDEN_DIM, dim_z)
        self.sigma_uc = nn.Linear(HIDDEN_DIM, dim_z)

    def forward(self, x: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = self.sigma_uc(out).exp()
        return dist.MultivariateNormal(mu, covariance_matrix=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(dim_z, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, dim_x),
        )
        self.sigma_uc = nn.Parameter(torch.as_tensor(0.0))

    def forward(self, x: torch.Tensor) -> dist.Distribution:
        return dist.Independent(dist.Normal(self.mu(x), scale=self.sigma_uc.exp() ** 0.5), 1)


def generate_toy_data(dataset: str, num_samples: int, seed: int | None):
    if seed is not None:
        torch.manual_seed(seed)

    if dataset == "anisotropic_gmm":
        k = torch.Size([3])
        centers = torch.as_tensor([[-2, 0], [2, 2], [2, -2]], dtype=torch.float32)
        widths = dist.Wishart(4, scale_tril=CLUSTER_STD ** 0.5 * torch.eye(2) / 2).sample(k)
        c = dist.Categorical(logits=torch.ones(k)).sample(torch.Size([num_samples]))
        x_c = dist.MultivariateNormal(centers, covariance_matrix=widths).sample(torch.Size([num_samples]))
        x = torch.gather(x_c, dim=1, index=c[:, None, None].repeat([1, 1, 2])).squeeze()

    elif dataset == "isotropic_gmm":
        k = torch.Size([3])
        centers = torch.as_tensor([[-2, 0], [2, 2], [2, -2]], dtype=torch.float32)
        widths = torch.eye(2)[None].repeat([3, 1, 1]) * CLUSTER_STD
        c = dist.Categorical(logits=torch.ones(k)).sample(torch.Size([num_samples]))
        x_c = dist.MultivariateNormal(centers, covariance_matrix=widths).sample(torch.Size([num_samples]))
        x = torch.gather(x_c, dim=1, index=c[:, None, None].repeat([1, 1, 2])).squeeze()

    elif dataset == "moons":
        num_samples = torch.Size([num_samples])
        pc = dist.Categorical(probs=torch.tensor([0.5, 0.5]))
        pz_c = dist.Normal(loc=torch.tensor([-torch.pi / 2, torch.pi / 2]), scale=CLUSTER_STD)
        c = pc.sample(num_samples)
        z = torch.gather(pz_c.sample(num_samples), dim=1, index=c[:, None]).squeeze()
        x = torch.stack([(c - 0.5) - torch.cos(z), (c - 0.5) / 2 - torch.sin(z)], dim=1)
        epsilon = dist.MultivariateNormal(torch.zeros(2), scale_tril=torch.eye(2) * 0.05).sample(num_samples)
        x = x + epsilon

    else:
        raise NotImplementedError

    return TensorDataset(x, c)


def collate(batch):
    batch = list(map(lambda b: {'x': b[0], 'c': b[1]}, batch))
    batch = merge_with(torch.vstack, batch)
    batch['c'] = batch['c'].squeeze()
    return batch


def fit_model(model_config: dict, data: TensorDataset, checkpoint_dir: str, seed: int | None):
    if seed is not None:
        torch.manual_seed(seed)

    dataloader = DataLoader(
        data,
        batch_size=len(data),
        collate_fn=collate,
        shuffle=True
    )

    model = VampPrior(
        dim_z=1,
        num_data=len(data),
        encoder=Encoder(dim_x=2, dim_z=1),
        decoder=Decoder(dim_x=2, dim_z=1),
        truncation_level=TRUNCATION_LEVEL,
        u_initial=next(iter(dataloader))['x'][:TRUNCATION_LEVEL],
        learning_rate=LEARNING_RATE,
        **model_config
    )

    trainer = get_trainer(
        early_stopping_metric='train_elbo',
        checkpoint_metrics=['train_elbo'],
        checkpoint_dir=checkpoint_dir,
        min_delta=0.01,
        patience=PATIENCE,
    )
    trainer.fit(model, train_dataloaders=dataloader)

    Path(f'{checkpoint_dir}/DONE').touch()

    torch.cuda.empty_cache()


def plot_results(data: TensorDataset, fig_path: str, model_path: str, seed: int | None):
    if seed is not None:
        torch.manual_seed(seed)

    x = data.tensors[0]

    model = VampPrior.load_from_checkpoint(model_path, map_location='cpu')

    cluster_probabilities = model.qc_z(model.log_pz_c(model.encoder(x).sample())).probs
    cluster_assignments = cluster_probabilities.argmax(dim=1)

    z_plot = torch.linspace(start=-100, end=100, steps=5000).unsqueeze(dim=1)
    pc_model = model.pc()
    pz_c_model = pc_model.probs * model.pz_c(z_plot)
    pz_model = pz_c_model.sum(dim=1)
    i_start = (pz_model > 1e-3).nonzero()[0]
    i_stop = (pz_model > 1e-3).nonzero()[-1]

    fig = plt.figure(figsize=(6, 10))
    gs = GridSpec(4, 2, figure=fig)

    ax = fig.add_subplot(gs[:2, :])
    ax.set_title('Model\'s Cluster Assignments')
    ax.set_xlabel('$x_1$')
    ax.set_xticks([])
    ax.set_ylabel('$x_2$')
    ax.set_yticks([])
    sns.scatterplot(
        x=x[:, 0],
        y=x[:, 1],
        hue=cluster_assignments,
        hue_order=range(model.k),
        palette=COLOR_PALETTE,
    )
    ax.get_legend().remove()

    ax = fig.add_subplot(gs[2, :])
    ax.set_title('Scaled Component Densities')
    ax.set_xlabel('z')
    ax.set_xticks([])
    ax.set_ylabel('density')
    ax.set_yticks([])
    sns.lineplot(pz_c_model[i_start:i_stop].detach(), alpha=0.5, dashes=False, palette=COLOR_PALETTE)
    ax.get_legend().remove()

    ax = fig.add_subplot(gs[3, :])
    ax.set_xlabel('z')
    ax.set_title('Mixture Density')
    ax.set_ylabel('density')
    ax.set_yticks([])
    ax.plot(z_plot[i_start:i_stop], pz_model[i_start:i_stop].detach(), alpha=0.75, color='black')

    plt.tight_layout()

    fig.savefig(fig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='moons')
    parser.add_argument('--num_samples', type=int, default=1_000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--replace_results', action='store_true')

    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    toy_data = generate_toy_data(args.dataset, args.num_samples, args.seed)

    for config in TEST_CASES:
        checkpoint_path = f'experiments/toy/{args.dataset}/{args.seed}/{str(config)}'
        os.makedirs(checkpoint_path, exist_ok=True)
        if not os.path.exists(f'{checkpoint_path}/DONE') or args.replace_results:
            fit_model(config, toy_data, checkpoint_path, args.seed)
        results_dir = f'results/toy/{args.dataset}/{args.seed}'
        os.makedirs(results_dir, exist_ok=True)
        plot_results(
            data=toy_data,
            fig_path=f'{results_dir}/{str(config)}.pdf',
            model_path=f'{checkpoint_path}/best_train_elbo.ckpt',
            seed=args.seed
        )
