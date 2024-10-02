import argparse
import anndata
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymde
import scanpy as sc
import scipy.sparse as sparse
import scvi
import seaborn as sns
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import zlib

from constants import COLOR_PALETTE, TEST_CASES
from cytoolz import merge_with
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


DATASET_CONFIG = {
    'brain large': dict(
        gene_likelihood='zinb',
        n_latent=10,
        n_layers=3,
        n_hidden=256,
    ),
    'cortex': dict(
        gene_likelihood='zinb',
        n_latent=10,
        n_layers=1,
        n_hidden=128,
    ),
    'lung atlas': dict(
        gene_likelihood='nb',
        n_latent=30,
        n_layers=2,
        n_hidden=128,
    ),
    'pbmc': dict(
        gene_likelihood='zinb',
        n_latent=10,
        n_layers=1,
        n_hidden=128,
    ),
    'retina': dict(
        gene_likelihood='zinb',
        n_latent=10,
        n_layers=1,
        n_hidden=128,
        learning_rate=5e-4
    ),
    'split-seq': dict(
        gene_likelihood='zinb',
        n_latent=30,
        n_layers=2,
        n_hidden=128,
    )
}
MAX_EPOCHS = 10_000
PATIENCE = 20


# class Encoder(nn.Module):
#     def __init__(self, dim_x: int, dim_z: int, dim_hidden: int, num_layers: int, dropout: float):
#         super().__init__()
#         dims = [dim_x] + [dim_hidden] * num_layers + [dim_z]
#         self.net = nn.Sequential()
#         for i in range(len(dims) - 1):
#             self.net.append(nn.Linear(dims[i], dims[i + 1]))
#             if i < len(dims) - 2:
#                 self.net.append(nn.BatchNorm1d(dims[i + 1]))
#                 self.net.append(nn.ReLU())
#                 self.net.append(nn.Dropout1d(dropout))
#
#     def forward(self, x: torch.Tensor, s: torch.Tensor) -> dist.MultivariateNormal:
#         xs = torch.concat([torch.log1p(x), s], dim=1)
#         out = self.net(xs)
#         mu = self.mu(out)
#         sigma = self.sigma(out)
#         return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))
#
#
# class Decoder(nn.Module):
#     def __init__(self, dim_x: int, dim_z: int, dim_hidden: int, num_layers: int, dropout: float):
#         super().__init__()
#         dims = [dim_z] + [dim_hidden] * num_layers + [2 * dim_x]
#         self.net = nn.Sequential()
#         for i in range(len(dims) - 1):
#             self.net.append(nn.Linear(dims[i], dims[i + 1]))
#             if i < len(dims) - 2:
#                 self.net.append(nn.BatchNorm1d(dims[i + 1]))
#                 self.net.append(nn.ReLU())
#                 self.net.append(nn.Dropout1d(dropout))
#         self.log_theta = nn.Parameter(torch.zeros(dim_x))
#
#     def forward(self, z: torch.Tensor, s: torch.Tensor, l: torch.Tensor) -> torch.Distribution:
#
#         torch.split(self.net())
#
#         return dist.Independent(dist.Normal(self.mu(z), F.softplus(self.log_sigma)), 1)


def load_dataset(dataset: str) -> anndata.AnnData:
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

    return scvi_dataset


def collate(batch, num_batches: int):
    batch = list(map(lambda b: {'x': b[0], 's': torch.LongTensor([b[1]]), 'c': torch.LongTensor([b[2]])}, batch))
    batch = merge_with(torch.vstack, batch)
    batch['x'] = (torch.flatten(batch['x'], start_dim=1) - 0.5) * 2
    batch['s'] = F.one_hot(batch['s'], num_classes=num_batches)
    batch['c'] = batch['c'].squeeze()
    return batch


def fit_scvi(batch_size: int, dataset: str, learning_rate: float, checkpoint_dir: str, seed: int | None):
    if seed is not None:
        torch.manual_seed(seed)

    scvi_dataset = load_dataset(dataset)

    scvi.model.SCVI.setup_anndata(scvi_dataset, layer='counts', batch_key='batch')
    model = scvi.model.SCVI(
        adata=scvi_dataset,
        deeply_inject_covariates=False,
        log_variational=True,
        **DATASET_CONFIG[dataset]
    )

    torch.set_float32_matmul_precision('high')
    model.train(
        batch_size=batch_size,
        check_val_every_n_epoch=1,
        early_stopping=True,
        early_stopping_patience=PATIENCE,
        max_epochs=MAX_EPOCHS,
        plan_kwargs=dict(
            lr=learning_rate,
            weight_decay=0.0,
            eps=1e-7,
            lr_patience=MAX_EPOCHS,
            n_steps_kl_warmup=0,
            n_epochs_kl_warmup=0,
        ),
        train_size=0.9,
    )

    np.save(f'{checkpoint_dir}/train_indices.npy', model.train_indices.copy())
    np.save(f'{checkpoint_dir}/valid_indices.npy', model.validation_indices.copy())

    pymde.seed(seed)
    embeddings = model.get_latent_representation()
    np.save(f'{checkpoint_dir}/embeddings.npy', embeddings)
    np.save(f'{checkpoint_dir}/mde.npy', scvi.model.utils.mde(embeddings))

    Path(f'{checkpoint_dir}/DONE').touch()

    torch.cuda.empty_cache()


    dataloader = DataLoader(
        tensor_dataset,
        batch_size=128,
        collate_fn=curry(collate, num_batches=scvi_dataset.obs['batch'].nunique()),
        shuffle=True
    )


def plot_mdes(dataset, checkpoint_dir, fig_path, fig_name):
    data = load_dataset(dataset)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))

    mde = np.load(f'{checkpoint_dir}/mde.npy')

    if os.path.exists(f'{checkpoint_dir}/cluster_probabilities.npy'):
        cluster_probabilities = np.load(f'{checkpoint_dir}/cluster_probabilities.npy')
        cluster_assignments = np.argmax(cluster_probabilities, axis=1)
        _, cluster_assignments = np.unique(cluster_assignments, return_inverse=True)
    else:
        cluster_assignments = np.zeros_like(data.obs['cell_type'])

    df = pd.DataFrame(data={
        'x': mde[:, 0],
        'y': mde[:, 1],
        'Batch': data.obs['batch'] + 1,
        'Cell type': data.obs['cell_type'],
        'Cluster': cluster_assignments + 1
    })

    # comparison plot
    kwargs = dict(
        s=3,
        alpha=0.5,
        palette=COLOR_PALETTE,
        rasterized=True,
    )
    sns.scatterplot(df, x='x', y='y', hue='Batch', ax=axes[0], legend='full', **kwargs)
    sns.scatterplot(df, x='x', y='y', hue='Cell type', ax=axes[1], legend='full', **kwargs)
    sns.scatterplot(df, x='x', y='y', hue='Cluster', style='Cluster', ax=axes[2], legend='full', **kwargs)
    for ax in axes:
        ax.set_xlabel('')
        ax.set_xticks([])
        ax.set_ylabel('')
        ax.set_yticks([])

    axes[0].set_title('Batch correction')
    axes[1].set_title('Bio conservation')
    axes[2].set_title('Model\'s Cluster Assignments')

    sns.move_legend(axes[0], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
    sns.move_legend(axes[1], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
    sns.move_legend(axes[2], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)

    plt.tight_layout()
    fig.savefig(f'{fig_path}/{fig_name}', dpi=200)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dataset', type=str, default='cortex')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--replace_results', action='store_true')
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    results = []
    for trial in range(1, args.num_trials + 1):
        trial_seed = zlib.crc32(f'{args.seed}{trial}'.encode())

        checkpoint_path = f'experiments/scRNA/{args.dataset}/{args.seed}/trial_{trial}/scVI'
        os.makedirs(checkpoint_path, exist_ok=True)
        if not os.path.exists(f'{checkpoint_path}/DONE') or args.replace_results:
            fit_scvi(args.batch_size, args.dataset, args.learning_rate, checkpoint_path, trial_seed)

        results_dir = f'results/scRNA/{args.dataset}/{args.seed}/trial_{trial}/scVI.pdf'
        os.makedirs(results_dir, exist_ok=True)
        plot_mdes(args.dataset, checkpoint_path, results_dir, 'scVI.pdf')

        # for config in TEST_CASES:
        #     checkpoint_path = f'experiments/scRNA/{args.dataset}/{args.seed}/trial_{trial}/{str(config)}'
        #     os.makedirs(checkpoint_path, exist_ok=True)
        #     if not os.path.exists(f'{checkpoint_path}/DONE') or args.replace_results:
        #         fit_model(config, args.dataset, checkpoint_path, trial_seed)
            # trial_results = test_model(config, args.dataset, checkpoint_path, trial_seed)
            # results.append({**trial_results, 'trial': trial})

    # results_dir = f'results/scRNA/{args.dataset}/{args.seed}'
    # os.makedirs(results_dir, exist_ok=True)
    # build_table(results, results_dir)
