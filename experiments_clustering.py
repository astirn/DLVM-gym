import argparse
import numpy as np
import os
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import zlib

from cytoolz import merge_with
from constants import TEST_CASES
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils import get_trainer
from vampprior_evolution import VampPrior

LEARNING_RATE = 5e-4
PATIENCE = 20
TRUNCATION_LEVEL = 50

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
        sigma = self.sigma_uc(out).exp()
        return dist.MultivariateNormal(mu, covariance_matrix=torch.diag_embed(sigma))


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
        self.sigma_uc = nn.Parameter(torch.as_tensor(0.0))

    def forward(self, x: torch.Tensor) -> dist.Distribution:
        return dist.Independent(dist.Normal(self.mu(x), scale=self.sigma_uc.exp() ** 0.5), 1)


def collate(batch):
    batch = list(map(lambda b: {'x': b[0], 'c': torch.LongTensor([b[1]])}, batch))
    batch = merge_with(torch.vstack, batch)
    batch['x'] = (torch.flatten(batch['x'], start_dim=1) - 0.5) * 2
    batch['c'] = batch['c'].squeeze()
    return batch


def fit_model(batch_size: int, model_config: dict, dataset: str, checkpoint_dir: str, seed: int):
    torch.manual_seed(seed)

    ds_train = getattr(datasets, dataset)(root='data', train=True, download=True, transform=ToTensor())
    ds_valid = getattr(datasets, dataset)(root='data', train=False, download=True, transform=ToTensor())

    dl_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate,
        'drop_last': True,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': False
    }
    dl_train = DataLoader(ds_train, shuffle=True, **dl_kwargs)
    dl_valid = DataLoader(ds_valid, shuffle=False, **dl_kwargs)

    data_dim = np.prod(ds_train[0][0].shape[1:])
    model = VampPrior(
        dim_x=data_dim,
        num_data=len(ds_train),
        encoder=Encoder(dim_x=data_dim, dim_z=model_config['dim_z']),
        decoder=Decoder(dim_x=data_dim, dim_z=model_config['dim_z']),
        u_constraint=F.tanh,
        truncation_level=TRUNCATION_LEVEL,
        learning_rate=LEARNING_RATE,
        **model_config
    )

    torch.set_float32_matmul_precision('high')
    trainer = get_trainer(
        checkpoint_metrics=['train_elbo', 'valid_elbo', 'nmi'],
        checkpoint_dir=checkpoint_dir,
        patience=PATIENCE,
    )
    trainer.fit(model, dl_train, dl_valid)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--latent_dim', type=int, default=30)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--replace_results', action='store_true')
    parser.add_argument('--seed', type=int, default=12345)

    args = parser.parse_args()

    for trial in range(1, args.num_trials + 1):
        trial_seed = zlib.crc32(f'{args.seed}{trial}'.encode())
        for config in TEST_CASES:
            config['dim_z'] = args.latent_dim
            save_path = f'experiments/clustering/{args.dataset}/trial_{trial}/'
            if not os.path.exists(save_path) or args.replace_results:
                fit_model(args.batch_size, config, args.dataset, save_path, trial_seed)
