import argparse
import numpy as np
import os
import pandas as pd
import torch
import torch.distributions as dist
import torch.nn as nn
import torchvision.datasets as datasets
import zlib

from cytoolz import merge_with
from constants import TABLE_COLUMNS, TEST_CASES
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from utils import get_trainer, seed_worker
from models import VampPrior

BATCH_SIZE = 1024
DIM_Z = 30
LEARNING_RATE = 5e-4
PATIENCE = 10
TRUNCATION_LEVEL = 100


class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int):
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

    def forward(self, batch: dict[str, torch.Tensor]) -> dist.MultivariateNormal:
        out = self.net(batch['x'])
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

    def forward(self, z: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return dist.Independent(dist.Normal(self.mu(z), scale=self.sigma_uc.exp() ** 0.5), 1).log_prob(batch['x'])


def load_dataset(dataset: str) -> (Dataset, Dataset):
    ds_train = getattr(datasets, dataset)(root='data', train=True, download=True, transform=ToTensor())
    ds_valid = getattr(datasets, dataset)(root='data', train=False, download=True, transform=ToTensor())
    return ds_train, ds_valid


def collate(batch):
    batch = list(map(lambda b: {'x': b[0], 'c': torch.LongTensor([b[1]])}, batch))
    batch = merge_with(torch.vstack, batch)
    batch['x'] = (torch.flatten(batch['x'], start_dim=1) - 0.5) * 2.0
    batch['c'] = batch['c'].squeeze()
    return batch


def get_dataloaders(ds_train: Dataset, ds_valid: Dataset, batch_size: int, seed: int | None):
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    dl_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate,
        'generator': g,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': True,
        'worker_init_fn': seed_worker if seed is not None else None,
    }

    return DataLoader(ds_train, shuffle=True, **dl_kwargs), DataLoader(ds_valid, shuffle=False, **dl_kwargs)


def fit_model(model_config: dict, dataset: str, checkpoint_dir: str, seed: int | None):
    if seed is not None:
        torch.manual_seed(seed)

    ds_train, ds_valid = load_dataset(dataset)
    dl_train, dl_valid = get_dataloaders(ds_train, ds_valid, BATCH_SIZE, seed)

    data_dim = np.prod(ds_train[0][0].shape[1:])
    model = VampPrior(
        dim_z=DIM_Z,
        num_data=len(ds_train),
        encoder=Encoder(dim_x=data_dim, dim_z=DIM_Z),
        decoder=Decoder(dim_x=data_dim, dim_z=DIM_Z),
        u_initial=next(iter(dl_train))['x'][:TRUNCATION_LEVEL],
        truncation_level=TRUNCATION_LEVEL,
        learning_rate=LEARNING_RATE,
        **model_config
    )

    trainer = get_trainer(
        early_stopping_metric='val_elbo',
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

    _, ds_test = get_dataloaders(*load_dataset(dataset), BATCH_SIZE, seed)

    trainer = get_trainer(checkpoint_metrics=[], checkpoint_dir=None, patience=0)

    model = VampPrior.load_from_checkpoint(f'{checkpoint_dir}/best_val_elbo.ckpt')
    model_results = trainer.test(model, ds_test)[0]

    model = VampPrior.load_from_checkpoint(f'{checkpoint_dir}/best_nmi.ckpt')
    clustering_results = trainer.validate(model, ds_test)[0]

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

    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--replace_results', action='store_true')
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    results = []
    for trial in range(1, args.num_trials + 1):
        trial_seed = zlib.crc32(f'{args.seed}{trial}'.encode())

        for config in TEST_CASES:
            checkpoint_path = f'experiments/clustering/{args.dataset}/{args.seed}/trial_{trial}/{str(config)}'
            os.makedirs(checkpoint_path, exist_ok=True)
            if not os.path.exists(f'{checkpoint_path}/DONE') or args.replace_results:
                Path(f'{checkpoint_path}/DONE').unlink(missing_ok=True)
                fit_model(config, args.dataset, checkpoint_path, trial_seed)
            trial_results = test_model(config, args.dataset, checkpoint_path, trial_seed)
            results.append({**trial_results, 'trial': trial})

    results_dir = f'results/clustering/{args.dataset}/{args.seed}'
    os.makedirs(results_dir, exist_ok=True)
    build_table(results, results_dir)
