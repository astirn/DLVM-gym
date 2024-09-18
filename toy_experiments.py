import argparse
import lightning
import lightning.pytorch.callbacks as callbacks
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from cytoolz import interleave, merge_with, pipe
from cytoolz.curried import partition
from matplotlib.gridspec import GridSpec
from sklearn import datasets
from vampprior_torch import VampPrior

HIDDEN_DIM = 100
TEST_CASES = {
    "VampPrior": dict(),
    "VampPrior with decoupled variance": dict(decoupled_variance=True),
    "VampPrior with Wish. priors": dict(wishart_prior=True),
    "VampPrior with Dir. priors": dict(dirichlet_prior=True),
    "VampPrior with Dir. + Wish. priors": dict(dirichlet_prior=True, wishart_prior=True),
    "VampPrior with DP priors": dict(dirichlet_process=True),
    "VampPrior with DP + Wish. priors": dict(dirichlet_process=True, wishart_prior=True)
}


class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x, HIDDEN_DIM),
            nn.Tanh(),
        )
        self.mu = nn.Linear(HIDDEN_DIM, dim_z)
        self.sigma = nn.Sequential(
            nn.Linear(HIDDEN_DIM, dim_z),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = self.sigma(out)
        return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(dim_z, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, dim_x),
        )
        self.log_sigma = nn.Parameter(torch.zeros(dim_x))

    def forward(self, x: torch.Tensor) -> dist.Distribution:
        return dist.Independent(dist.Normal(self.mu(x), F.softplus(self.log_sigma)), 1)


def collate(batch):
    batch = list(map(lambda b: {'x': b[0], 'y': b[1]}, batch))
    batch = merge_with(torch.vstack, batch)
    batch['y'] = batch['y'].squeeze()
    return batch


def generate_toy_data(dataset: str, num_samples: int, seed: int | None):
    if seed is not None:
        torch.manual_seed(seed)

    if dataset == "blobs":
        dataset = datasets.make_blobs(n_samples=num_samples, random_state=seed)
        x, c = map(torch.Tensor, dataset)
    elif dataset == "circles":
        dataset = datasets.make_circles(n_samples=num_samples, factor=0.5, noise=0.05, random_state=seed)
        x, c = map(torch.Tensor, dataset)
    elif dataset == "custom":
        num_samples = torch.Size([num_samples])
        pc = dist.Categorical(probs=torch.tensor([0.5, 0.5]))
        pz_c = dist.Normal(loc=torch.tensor([-torch.pi / 2, torch.pi / 2]), scale=0.25)
        c = pc.sample(num_samples)
        z = torch.gather(pz_c.sample(num_samples), dim=1, index=c[:, None]).squeeze()
        x = torch.stack([torch.cos(z) + (c - 0.5), torch.sin(z) - (c - 0.5)], dim=1)
        epsilon = dist.MultivariateNormal(torch.zeros(2), scale_tril=torch.eye(2) * 2e-2).sample(num_samples)
        x = x + epsilon
    elif dataset == "moons":
        dataset = datasets.make_moons(n_samples=num_samples, noise=0.05, random_state=seed)
        x, c = map(torch.Tensor, dataset)
    else:
        raise NotImplementedError

    return data.TensorDataset(x, c)


def fit_models(dataset: data.TensorDataset, base_dir: str, replace_results: bool, seed: int | None):
    for name, config in TEST_CASES.items():
        if seed is not None:
            torch.manual_seed(seed)

        checkpoint_file = f'{base_dir}/{name}/model.ckpt'
        if not os.path.exists(checkpoint_file) or replace_results:

            dataloader = data.DataLoader(
                dataset,
                batch_size=len(dataset),
                collate_fn=collate,
                shuffle=True
            )

            model = VampPrior(
                dim_x=2,
                dim_z=1,
                num_data=len(dataset),
                encoder=Encoder,
                decoder=Decoder,
                truncation_level=100,
                learning_rate=5e-4,
                **config
            )

            trainer = lightning.Trainer(
                callbacks=[callbacks.EarlyStopping(
                    min_delta=0.01,
                    monitor='train_elbo',
                    mode='max',
                    patience=500,
                    verbose=True
                )],
                devices=1,
                enable_checkpointing=False,
                logger=False,
                max_epochs=-1,
                num_sanity_val_steps=0,
            )
            trainer.fit(model, train_dataloaders=dataloader)
            trainer.save_checkpoint(checkpoint_file)

        torch.cuda.empty_cache()


def plot_results(dataset: data.TensorDataset, base_dir: str, seed: int | None):
    for name, config in TEST_CASES.items():
        if seed is not None:
            torch.manual_seed(seed)

        x = dataset.tensors[0]

        checkpoint_file = f'{base_dir}/{name}/model.ckpt'
        if os.path.exists(checkpoint_file):

            model = VampPrior.load_from_checkpoint(checkpoint_file)

            cluster_probabilities = model.qc_z(model.log_pz_c(model.qz_x(x).sample())).probs
            cluster_assignments = cluster_probabilities.argmax(dim=1)

            z_plot = torch.linspace(start=-30, end=30, steps=5000).unsqueeze(dim=1)
            pc_model = model.pc()
            pz_c_model = pc_model.probs * model.pz_c(z_plot)
            pz_model = pz_c_model.sum(dim=1)
            i_start = (pz_model > 1e-5).nonzero()[0]
            i_stop = (pz_model > 1e-5).nonzero()[-1]

            fig = plt.figure(figsize=(6, 10))
            fig.suptitle(name)
            gs = GridSpec(4, 2, figure=fig)

            palette = sns.color_palette('tab20c', n_colors=20) + sns.color_palette('tab20b', n_colors=20)
            palette = pipe(palette, partition(4), interleave, list)

            ax = fig.add_subplot(gs[0, :])
            ax.set_xlabel('z')
            ax.set_xticks([])
            ax.set_title('Mixture Density')
            ax.set_ylabel('density')
            ax.set_yticks([])
            ax.plot(pz_model[i_start:i_stop].detach(), alpha=0.75, color='black')

            ax = fig.add_subplot(gs[1, :])
            ax.set_title('Scaled Component Densities')
            ax.set_xlabel('z')
            ax.set_xticks([])
            ax.set_ylabel('density')
            ax.set_yticks([])
            sns.lineplot(pz_c_model[i_start:i_stop].detach(), alpha=0.5, dashes=False, palette=palette)
            ax.get_legend().remove()

            ax = fig.add_subplot(gs[2:, :])
            ax.set_title('Data Cluster Assignments')
            ax.set_xlabel('$x_1$')
            ax.set_xticks([])
            ax.set_ylabel('$x_2$')
            ax.set_yticks([])
            sns.scatterplot(
                x=x[:, 0],
                y=x[:, 1],
                hue=cluster_assignments,
                hue_order=range(model.k),
                palette=palette,
            )
            ax.get_legend().remove()

            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='custom')
    parser.add_argument('--num_samples', type=int, default=1_000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--replace_results', action='store_true')

    args = parser.parse_args()

    experiment_dir = f'experiments/toy/{args.dataset}'
    os.makedirs(experiment_dir, exist_ok=True)

    if args.seed is not None:
        torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')

    toy_data = generate_toy_data(args.dataset, args.num_samples, args.seed)

    fit_models(toy_data, experiment_dir, args.replace_results, args.seed)

    plot_results(toy_data, experiment_dir, args.seed)
