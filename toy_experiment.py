import argparse
import os
import lightning
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import matplotlib.pyplot as plt
import seaborn as sns

from cytoolz import merge_with
from vampprior_torch import VampPrior


class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_x, 10), nn.Tanh())
        self.mu = nn.Linear(10, dim_z)
        self.sigma = nn.Sequential(nn.Linear(10, dim_z), nn.Softplus())

    def forward(self, x: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = self.sigma(out)
        return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(dim_z, 10), nn.Tanh(), nn.Linear(10, dim_x))
        self.log_sigma = nn.Parameter(torch.zeros(dim_x))

    def forward(self, x: torch.Tensor) -> dist.Distribution:
        return dist.Independent(dist.Normal(self.mu(x), F.softplus(self.log_sigma)), 1)


def collate(batch):
    batch = list(map(lambda b: {'x': b[0], 'y': b[1]}, batch))
    batch = merge_with(torch.vstack, batch)
    batch['y'] = batch['y'].squeeze()
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_epochs', type=int, default=5_000)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--truncation_level', type=int, default=100)

    args = parser.parse_args()

    # # make experiment directory
    # exp_path = os.path.join('experiments', 'decoupling')
    # os.makedirs(exp_path, exist_ok=True)

    torch.set_float32_matmul_precision('high')

    # sample data from generative process
    pc = dist.Categorical(probs=torch.tensor([0.5, 0.5]))
    pz_c = dist.Normal(loc=torch.tensor([-torch.pi / 2, torch.pi / 2]), scale=0.25)
    pz = dist.MixtureSameFamily(dist.Categorical(probs=torch.tensor([0.5, 0.5])), pz_c)
    c = pc.sample(sample_shape=torch.Size([args.num_samples]))
    z = torch.gather(pz_c.sample(sample_shape=torch.Size([args.num_samples])), dim=1, index=c[:, None]).squeeze()
    x = torch.stack([torch.cos(z) + (c - 0.5), torch.sin(z) - (c - 0.5)], dim=1)
    epsilon = dist.MultivariateNormal(torch.zeros(2), scale_tril=torch.eye(2) * 2e-2).sample(sample_shape=torch.Size([args.num_samples]))
    x = x + epsilon

    dataloader = data.DataLoader(
        data.TensorDataset(x, c),
        batch_size=args.num_samples,
        collate_fn=collate,
        shuffle=True
    )

    # test cases
    common_kwargs = dict(
        dim_x=x.shape[1],
        dim_z=z.shape[0],
        num_data=args.num_samples,
        encoder=Encoder,
        decoder=Decoder,
        truncation_level=args.truncation_level,
        learning_rate=args.learning_rate,
        u_initial=next(iter(dataloader))['x'][:args.truncation_level],
    )
    test_cases = {
        "VampPrior": VampPrior(
            **common_kwargs
        ),
        "1": VampPrior(
            kl_warmup=1_000,
            dirichlet_prior=True,
            wishart_prior=True,
            **common_kwargs
        ),
        "2": VampPrior(
            kl_warmup=1_000,
            dirichlet_process=True,
            wishart_prior=True,
            **common_kwargs
        ),
    }

    for name, model, in test_cases.items():
        if args.seed is not None:
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(args.seed)

        trainer = lightning.Trainer(
            devices=1,
            enable_checkpointing=False,
            logger=False,
            max_epochs=args.num_epochs,
            num_sanity_val_steps=0,
        )
        trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)

        cluster_assignments = model.qc_z(model.log_pz_c(model.qz_x(x).sample())).probs.argmax(dim=1)
        utilized_clusters = cluster_assignments.unique()
        pz_c_model = model.qz_x(model.u_uc)
        pz_model = dist.MixtureSameFamily(model.pc(), pz_c_model)
        cluster_means = pz_c_model.mean[utilized_clusters].squeeze()
        cluster_stds = pz_c_model.stddev[utilized_clusters].squeeze()
        i_max = cluster_means.argmax()
        i_min = cluster_means.argmin()
        z_plot = torch.linspace(
            start=cluster_means[i_min] - 3 * cluster_stds[i_min],
            end=cluster_means[i_max] + 3 * cluster_stds[i_max],
            steps=1000
        ).unsqueeze(dim=1)

        # plot p(z|c) and p(z)
        fig, ax = plt.subplots(nrows=3, figsize=(12, 12))
        ax[0].set_title(f'{name} Component Densities')
        ax[0].plot(z_plot, pz_c_model.log_prob(z_plot[:, None])[:, utilized_clusters].exp().detach())
        ax[1].set_xlabel('z')
        ax[0].set_ylabel('density')
        ax[1].set_title(f'{name} Mixture density')
        ax[1].plot(z_plot, pz_model.log_prob(z_plot).exp().detach())
        ax[1].set_xlabel('z')
        ax[1].set_ylabel('density')
        sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=np.unique(cluster_assignments, return_inverse=True)[1], ax=ax[2])
        plt.tight_layout()
        plt.show()

        torch.cuda.empty_cache()
