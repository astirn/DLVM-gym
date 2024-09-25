import argparse
import anndata
import lightning
import lightning.pytorch.callbacks as callbacks
import numpy as np
import os
import pandas as pd
import pymde
import scanpy as sc
import scipy.sparse as sparse
import scvi
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import zlib

from cytoolz import merge_with
from cytoolz.curried import curry
from torch.utils.data import DataLoader, TensorDataset
from typing import Literal
from vampprior_evolution import VampPrior


class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, dim_hidden: int, num_layers: int, dropout: float):
        super().__init__()
        dims = [dim_x] + [dim_hidden] * num_layers + [dim_z]
        self.net = nn.Sequential()
        for i in range(len(dims) - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.net.append(nn.BatchNorm1d(dims[i + 1]))
                self.net.append(nn.ReLU())
                self.net.append(nn.Dropout1d(dropout))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> dist.MultivariateNormal:
        xs = torch.concat([torch.log1p(x), s], dim=1)
        out = self.net(xs)
        mu = self.mu(out)
        sigma = self.sigma(out)
        return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, dim_hidden: int, num_layers: int, dropout: float):
        super().__init__()
        dims = [dim_z] + [dim_hidden] * num_layers + [2 * dim_x]
        self.net = nn.Sequential()
        for i in range(len(dims) - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                self.net.append(nn.BatchNorm1d(dims[i + 1]))
                self.net.append(nn.ReLU())
                self.net.append(nn.Dropout1d(dropout))
        self.log_theta = nn.Parameter(torch.zeros(dim_x))

    def forward(self, z: torch.Tensor, s: torch.Tensor, l: torch.Tensor) -> torch.Distribution:

        torch.split(self.net())

        return dist.Independent(dist.Normal(self.mu(x), F.softplus(self.log_sigma)), 1)


class SingleCellVampPrior(VampPrior):
    def __init__(
            self,
            dim_z: int,
            num_batches: int,
            num_genes: int,
            encoder: callable,
            decoder: callable,
            hidden_dims: list[int],
            likelihood: Literal['zinb', 'nb'] = 'zinb',
            **kwargs,
    ):
        VampPrior.__init__(
            self,
            dim_x=num_batches + num_genes,
            dim_z=dim_z,
            encoder=curry(encoder)(hidden_dims=hidden_dims),
            decoder=curry(decoder)(hidden_dims=hidden_dims, likelihood=likelihood),
            **kwargs
        )
        self.log_px_zsl = VampPrior.px_z  # renamed since decoder r

    def evidence_lower_bound(self, x: torch.Tensor, s: torch.Tensor) -> (torch.Tensor, dict[str, dist.Distribution]):
        qz_x = self.qz_x(x, s)
        z = qz_x.rsample()

        log_pz_c = self.log_pz_c(z)
        qc_z = self.qc_z(log_pz_c)

        elbo_locals = (
            + self.log_px_zl(x, z, x.sum(dim=1)).sum()
            - self.dkl_z(qz_x, qc_z, log_pz_c)
            - self.dkl_c(qc_z)  # zero only for original VampPrior
        ) / x.shape[0]
        elbo_globals = (
            - self.dkl_pi()  # zero unless Dirichlet prior
            - self.dkl_v()  # zero unless Dirichlet process
            - self.dkl_Lambda()  # zero unless Wishart prior
        ) / self.num_data
        elbo = elbo_locals + elbo_globals
        self.kl_warmup = max(self.kl_warmup - 1, 0)

        return elbo, {'qz_x': qz_x, 'qc_z': qc_z}

    def training_step(self, batch, batch_idx):
        elbo, _ = self.evidence_lower_bound(batch['x'], batch['s'])
        self.log("train_elbo", elbo, prog_bar=True)
        return elbo

    def validation_step(self, batch, batch_idx):
        elbo, variational_family = self.evidence_lower_bound(batch['x'], batch['s'])
        self.log("val_elbo", elbo, on_epoch=True, prog_bar=True, batch_size=batch['x'].shape[0])
        self.clustering_metrics.update(**{**batch, **variational_family})

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.clustering_metrics.compute(), on_epoch=True, prog_bar=True)
        self.clustering_metrics.reset()


def dataset_config(dataset: str):
    if dataset == 'brain large':
        return dict(
            dim_z=10,
            hidden_dims=[256, 256, 256],
            likelihood='zinb',
            learning_rate=1e-3
        )
    elif dataset == 'cortex':
        return dict(
            dim_z=10,
            hidden_dims=[128],
            likelihood='zinb',
            learning_rate=4e-4
        )
    elif dataset == 'lung atlas':
        return dict(
            dim_z=30,
            hidden_dims=[128, 128],
            likelihood='nb',
            learning_rate=1e-3
        )
    elif dataset == 'pbmc':
        return dict(
            dim_z=10,
            hidden_dims=[128],
            likelihood='zinb',
            learning_rate=4e-4
        )
    elif dataset == 'retina':
        return dict(
            dim_z=10,
            hidden_dims=[128],
            likelihood='zinb',
            learning_rate=5e-4
        )
    elif dataset == 'split-seq':
        return dict(
            dim_z=30,
            hidden_dims=[128, 128],
            likelihood='zinb',
            learning_rate=1e-3
        )
    else:
        raise NotImplementedError


def load_dataset(dataset: str) -> (anndata.AnnData, TensorDataset):
    if dataset == 'brain large':
        scvi_dataset = scvi.data.brainlarge_dataset()
        scvi_dataset.obs.rename(columns={'labels': 'cell_type'}, inplace=True)
    elif dataset == 'cortex':
        scvi_dataset = scvi.data.cortex()
        sc.pp.highly_variable_genes(scvi_dataset, n_top_genes=558, flavor='seurat_v3', subset=True)
        scvi_dataset.obs['batch'] = 0
    elif dataset == 'pbmc':
        scvi_dataset = scvi.data.pbmc_dataset()
        scvi_dataset.obs.rename(columns={'str_labels': 'cell_type'}, inplace=True)
    elif dataset == 'lung atlas':
        scvi_dataset = sc.read(
            'data/lung_atlas.h5ad',
            backup_url="https://figshare.com/ndownloader/files/24539942",
        )
        sc.pp.highly_variable_genes(scvi_dataset, n_top_genes=2000, flavor='seurat_v3', subset=True, batch_key='batch')
        _, scvi_dataset.obs['batch'] = np.unique(scvi_dataset.obs['batch'], return_inverse=True)
    elif dataset == 'retina':
        scvi_dataset = scvi.data.retina()
        scvi_dataset.obs.rename(columns={'labels': 'cell_type'}, inplace=True)
    elif dataset == 'split-seq':
        meta = pd.read_csv('data/split-seq/cells_meta.tsv.gz', sep='\t')
        meta.rename(columns={'polydT': 'batch', 'cellchat_clusters': 'cell_type'}, inplace=True)
        scvi_dataset = anndata.AnnData(
            X=sparse.load_npz('data/split-seq/counts_small.npz'),
            obs=meta[['batch', 'cell_type']]
        )
    else:
        raise NotImplementedError
    sc.pp.filter_genes(scvi_dataset, min_counts=3)
    sc.pp.filter_cells(scvi_dataset, min_counts=3)
    scvi_dataset.layers['counts'] = scvi_dataset.X.todense() if sparse.issparse(scvi_dataset.X) else scvi_dataset.X.copy()
    scvi_dataset.obs['batch'] = scvi_dataset.obs['batch'].astype(int)

    tensor_dataset = TensorDataset(
        torch.tensor(scvi_dataset.layers['counts'], dtype=torch.float32),
        torch.tensor(scvi_dataset.obs['batch'], dtype=torch.float32),
        torch.tensor(scvi_dataset.obs['cell_type'], dtype=torch.float32),
    )

    return scvi_dataset, tensor_dataset


def collate(batch, num_batches: int):
    batch = list(map(lambda b: {'x': b[0], 's': torch.LongTensor([b[1]]), 'c': torch.LongTensor([b[2]])}, batch))
    batch = merge_with(torch.vstack, batch)
    batch['x'] = (torch.flatten(batch['x'], start_dim=1) - 0.5) * 2
    batch['s'] = F.one_hot(batch['s'], num_classes=num_batches)
    batch['c'] = batch['c'].squeeze()
    return batch


def fit_models(dataset: str, trial_dir: str, seed: int):
    scvi_dataset, tensor_dataset = load_dataset(dataset)
    config = dataset_config(dataset)

    # fit scVI
    torch.manual_seed(seed)
    scvi.model.SCVI.setup_anndata(scvi_dataset, layer='counts', batch_key='batch')
    model = scvi.model.SCVI(
        adata=scvi_dataset,
        n_hidden=config['hidden_dim'],
        n_latent=config['latent_dim'],
        n_layers=config['n_layers'],
        gene_likelihood=config['likelihood'],
        deeply_inject_covariates=False,
        log_variational=True)
    model.train(
        max_epochs=args.max_epochs,
        train_size=args.train_ratio,
        batch_size=config['batch_size'],
        check_val_every_n_epoch=1,
        early_stopping=True,
        early_stopping_patience=args.patience,
        plan_kwargs=dict(
            lr=config['learning_rate'],
            weight_decay=0.0,
            eps=1e-7,
            lr_patience=args.max_epochs,
            n_steps_kl_warmup=0,
            n_epochs_kl_warmup=0
        )
    )

    # save scVI results
    i_train = model.train_indices.copy()
    np.save(f'{trial_dir}/scVI/train_indices.npy', i_train)
    i_valid = model.validation_indices.copy()
    np.save(f'{trial_dir}/scVI/valid_indices.npy', i_valid)
    pymde.seed(trial_seed)
    embeddings = model.get_latent_representation()
    np.save(f'{trial_dir}/scVI/embeddings.npy', embeddings)
    mde = scvi.model.utils.mde(embeddings)
    np.save(f'{trial_dir}/scVI/mde.npy', mde)


    model = SingleCellVampPrior(
        num_batches=scvi_dataset.obs['batch'].nunique(),
        num_genes=scvi_dataset.n_vars,
        **dataset_config
    )

    dataloader = DataLoader(
        tensor_dataset,
        batch_size=128,
        collate_fn=curry(collate, num_batches=scvi_dataset.obs['batch'].nunique()),
        shuffle=True
    )

    trainer = lightning.Trainer(
        max_epochs=100,
        gpus=1,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_elbo', patience=10, mode='max'),
            callbacks.ModelCheckpoint(monitor='val_elbo', mode='max', dirpath=trial_dir, filename='model')
        ]
    )

    trainer.fit(model, dataloader)

    model.save_hyperparameters(f'{trial_dir}/hparams.yaml')

    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='custom')
    parser.add_argument('--num_samples', type=int, default=1_000)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--replace_results', action='store_true')

    args = parser.parse_args()

    experiment_dir = f'experiments/scRNA/{args.dataset}/{args.seed}'
    os.makedirs(experiment_dir, exist_ok=True)
    # results_dir = f'results/scRNA/{args.dataset}'
    # os.makedirs(results_dir, exist_ok=True)

    if args.seed is not None:
        torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')

    for t in range(1, args.num_trials + 1):
        print(f'*** {args.dataset} | Trial {t + 1}')
        trial_seed = int(zlib.crc32(str(args.trial * (args.seed or 1)).encode())) % (2 ** 32 - 1)

        fit_models(args.dataset, f'{experiment_dir}/trial_{t + 1}', trial_seed)
