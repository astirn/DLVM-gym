import argparse
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets

from cytoolz import merge_with
from constants import MIN_STD, TEST_CASES
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils import get_trainer, softplus_inverse
from vampprior_evolution import VampPrior


class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int,):
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
        self.sigma_uc = nn.Linear(2000, dim_z)

    def forward(self, x: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = torch.clamp_min(F.softplus(self.sigma_uc(out)), MIN_STD)
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
            nn.Linear(500, dim_x)
        )
        self.sigma_uc = nn.Parameter(torch.zeros(dim_x))

    def forward(self, x: torch.Tensor) -> dist.Distribution:
        sigma = torch.clamp_min(F.softplus(self.sigma_uc), MIN_STD)
        return dist.Independent(dist.Normal(self.mu(x), sigma), 1)


def collate(batch):
    batch = list(map(lambda b: {'x': b[0], 'c': torch.LongTensor([b[1]])}, batch))
    batch = merge_with(torch.vstack, batch)
    batch['x'] = (torch.flatten(batch['x'], start_dim=1) - 0.5) * 2
    batch['c'] = batch['c'].squeeze()
    return batch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--latent_dim', type=int, default=30)
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--categorical_prior', action='store_true', default=False)
    parser.add_argument('--decoupled_variance', action='store_true', default=False)
    parser.add_argument('--dirichlet_prior', action='store_true', default=False)
    parser.add_argument('--dirichlet_process', action='store_true', default=False)
    parser.add_argument('--wishart_prior', action='store_true', default=False)

    args = parser.parse_args()

    # random seed
    if args.seed is not None:
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)

    # load dataset
    dataset = getattr(datasets, args.dataset)
    ds_train = dataset(root='data', train=True, download=True, transform=ToTensor())
    ds_test = dataset(root='data', train=False, download=True, transform=ToTensor())

    # create dataloaders
    dl_kwargs = {
        'batch_size': args.batch_size,
        'collate_fn': collate,
        'drop_last': True,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': False
    }
    dl_train = torch.utils.data.DataLoader(ds_train, shuffle=True, **dl_kwargs)
    dl_test = torch.utils.data.DataLoader(ds_test, shuffle=False, **dl_kwargs)

    # build model
    data_dim = np.prod(ds_train[0][0].shape[1:])
    model = VampPrior(
        dim_x=data_dim,
        dim_z=args.latent_dim,
        num_data=len(ds_train),
        encoder=Encoder(dim_x=data_dim, dim_z=args.latent_dim),
        decoder=Decoder(dim_x=data_dim, dim_z=args.latent_dim),
        u_constraint=F.tanh,
        truncation_level=50,
        learning_rate=5e-4,
        categorical_prior=args.categorical_prior,
        decoupled_variance=args.decoupled_variance,
        dirichlet_prior=args.dirichlet_prior,
        dirichlet_process=args.dirichlet_process,
        dirichlet_process_variational_family=True,
        wishart_prior=args.wishart_prior,
    )

    # train model
    torch.set_float32_matmul_precision('high')
    trainer = get_trainer(checkpoint_metrics=[], checkpoint_dir=None, patience=20)
    trainer.fit(model, dl_train, dl_test)
