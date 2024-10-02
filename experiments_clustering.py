import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import zlib

from cytoolz import merge_with
from constants import TABLE_COLUMNS, TEST_CASES
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils import get_trainer
from models import VampPrior

DIM_Z = 30
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


def fit_model(batch_size: int, model_config: dict, dataset: str, learning_rate: float, checkpoint_dir: str, seed: int | None):
    if seed is not None:
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
        dim_z=DIM_Z,
        num_data=len(ds_train),
        encoder=Encoder(dim_x=data_dim, dim_z=DIM_Z),
        decoder=Decoder(dim_x=data_dim, dim_z=DIM_Z),
        u_constraint=F.tanh,
        truncation_level=TRUNCATION_LEVEL,
        learning_rate=learning_rate,
        **model_config
    )

    torch.set_float32_matmul_precision('high')
    trainer = get_trainer(
        checkpoint_metrics=['train_elbo', 'val_elbo', 'nmi'],
        checkpoint_dir=checkpoint_dir,
        patience=PATIENCE,
    )
    trainer.fit(model, dl_train, dl_valid)

    Path(f'{checkpoint_dir}/DONE').touch()

    torch.cuda.empty_cache()


def test_model(model_config: dict, dataset: str, checkpoint_dir: str, seed: int | None):
    if seed is not None:
        torch.manual_seed(seed)

    ds_test = getattr(datasets, dataset)(root='data', train=False, download=True, transform=ToTensor())
    dl_test = DataLoader(ds_test, batch_size=len(ds_test), collate_fn=collate, shuffle=False)

    trainer = get_trainer(checkpoint_metrics=[], checkpoint_dir=None, patience=0)

    model = VampPrior.load_from_checkpoint(f'{checkpoint_dir}/best_val_elbo.ckpt')
    model_results = trainer.test(model, dl_test)[0]

    model = VampPrior.load_from_checkpoint(f'{checkpoint_dir}/best_nmi.ckpt')
    clustering_results = trainer.validate(model, dl_test)[0]

    torch.cuda.empty_cache()

    return {**model_config, **model_results, **clustering_results}


def build_table(experimental_results: list[dict], save_path: str):
    df = pd.DataFrame(experimental_results)
    df.rename(columns=TABLE_COLUMNS, inplace=True)
    df.fillna(False, inplace=True)
    df = df[TABLE_COLUMNS.values()]
    index = list(TABLE_COLUMNS.values())
    index = index[:index.index('log p(x)')]
    df = df.groupby(index, sort=False).mean().reset_index()
    df.to_pickle(f'{save_path}/results.pkl')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--replace_results', action='store_true')
    parser.add_argument('--seed', type=int, default=4282023)

    args = parser.parse_args()

    results = []
    for trial in range(1, args.num_trials + 1):
        trial_seed = zlib.crc32(f'{args.seed}{trial}'.encode())

        for config in TEST_CASES:
            checkpoint_path = f'experiments/clustering/{args.dataset}/{args.seed}/trial_{trial}/{str(config)}'
            os.makedirs(checkpoint_path, exist_ok=True)
            if not os.path.exists(f'{checkpoint_path}/DONE') or args.replace_results:
                Path(f'{checkpoint_path}/DONE').unlink(missing_ok=True)
                fit_model(args.batch_size, config, args.dataset, args.learning_rate, checkpoint_path, trial_seed)
            trial_results = test_model(config, args.dataset, checkpoint_path, trial_seed)
            results.append({**trial_results, 'trial': trial})

    results_dir = f'results/clustering/{args.dataset}/{args.seed}'
    os.makedirs(results_dir, exist_ok=True)
    build_table(results, results_dir)
