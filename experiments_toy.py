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
    def __init__(self, dim_x: int, dim_z: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x, HIDDEN_DIM),
            nn.Tanh(),
        )
        self.mu = nn.Linear(HIDDEN_DIM, dim_z)
        self.sigma_uc = nn.Linear(HIDDEN_DIM, dim_z)

    def forward(self, batch: dict[str, torch.Tensor]) -> dist.MultivariateNormal:
        out = self.net(batch['x'])
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

    def forward(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return dist.Independent(dist.Normal(self.mu(z), scale=self.sigma_uc.exp() ** 0.5), 1).log_prob(batch['x'])


def generate_data(dataset: str, num_samples: int, seed: int | None):
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


def get_dataloader(dataset: TensorDataset, seed: int | None) -> DataLoader:
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    return DataLoader(
        dataset,
        batch_size=len(dataset),
        collate_fn=collate,
        generator=g,
        shuffle=True
    )


def fit_model(model_config: dict, dataset: TensorDataset, checkpoint_dir: str, seed: int | None):
    if seed is not None:
        torch.manual_seed(seed)

    dataloader = get_dataloader(dataset, seed)

    model = VampPrior(
        dim_z=1,
        num_data=len(dataset),
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


def plot_results(dataset: TensorDataset, checkpoint_dir: str, fig_path: str, seed: int | None):
    if seed is not None:
        torch.manual_seed(seed)

    trainer = get_trainer(checkpoint_metrics=[], checkpoint_dir=None, patience=0)

    model = VampPrior.load_from_checkpoint(f"{checkpoint_dir}/best_train_elbo.ckpt")
    predictions = trainer.predict(model, get_dataloader(dataset, seed))
    predictions = merge_with(lambda x: torch.vstack(x).cpu().numpy(), predictions)
    cluster_assignments = predictions['cluster_probabilities'].argmax(axis=1)

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
        x=dataset.tensors[0][:, 0],
        y=dataset.tensors[0][:, 1],
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

    toy_data = f'experiments/toy/{args.dataset}/{args.seed}/toy_data.pt'
    if not os.path.exists(toy_data):
        torch.save(
            generate_data(args.dataset, args.num_samples, args.seed),
            toy_data
        )
    toy_data = torch.load(toy_data)

    results_dir = f'results/toy/{args.dataset}/{args.seed}'
    os.makedirs(results_dir, exist_ok=True)

    for config in TEST_CASES:
        config_dir = f'experiments/toy/{args.dataset}/{args.seed}/{str(config)}'
        os.makedirs(config_dir, exist_ok=True)

        if not os.path.exists(f'{config_dir}/DONE') or args.replace_results:
            fit_model(config, toy_data, config_dir, args.seed)

        plot_results(
            dataset=toy_data,
            checkpoint_dir=config_dir,
            fig_path=f'{results_dir}/{str(config)}.pdf',
            seed=args.seed
        )
