import argparse
import anndata
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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
from cytoolz import concat, merge_with
from cytoolz.curried import curry
from models import SingleCellVampPrior
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from utils import get_trainer, seed_worker
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
TRUNCATION_LEVEL = 50


class Encoder(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_layers: int, n_hidden: int, **kwargs):
        super().__init__()
        dims = [dim_in] + [n_hidden] * n_layers
        self.net = nn.Sequential(*list(concat(
            [[
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1], momentum=0.01, eps=0.001),
                nn.ReLU(),
                nn.Dropout(0.1),
            ] for i in range(len(dims) - 1)]
        )))
        self.mu = nn.Linear(n_hidden, dim_out)
        self.log_sigma = nn.Linear(n_hidden, dim_out)

    def forward(self, xs: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(xs)
        mu = self.mu(out)
        sigma = (self.log_sigma(out).exp() + 1e-4).sqrt()
        return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, gene_likelihood: str, n_layers: int, n_hidden: int, **kwargs):
        super().__init__()
        dims = [dim_in] + [n_hidden] * n_layers
        self.net = nn.Sequential(*list(concat(
            [[
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1], momentum=0.01, eps=0.001),
                nn.ReLU(),
            ] for i in range(len(dims) - 1)]
        )))
        self.gene_likelihood = gene_likelihood
        self.log_theta = nn.Parameter(torch.zeros(dim_out))
        self.fw = nn.Sequential(
            nn.Linear(dims[-1], dim_out),
            nn.Softmax(dim=1),
        )
        self.fh = nn.Linear(dims[-1], dim_out)

    def forward(self, x: torch.Tensor, z: torch.Tensor, s: torch.Tensor, eps=1e-8):
        out = self.net(torch.concat([z, s], dim=1))
        mu = self.fw(out) * x.sum(dim=1, keepdim=True)
        theta = self.log_theta.exp()
        log_p = (mu + eps).log() - (mu + theta + eps).log()
        log_1_minus_p = (theta + eps).log() - (mu + theta + eps).log()
        if self.gene_likelihood == 'nb':
            log_prob = (
                + x * log_p
                + theta * log_1_minus_p
                + (x + theta).lgamma()
                - theta.lgamma()
                - (x + 1).lgamma()
            )
        elif self.gene_likelihood == 'zinb':
            fh = self.fh(out)
            zero_ll = F.softplus(-fh + theta * log_1_minus_p) - F.softplus(-fh)
            non_zero_ll = (
                - fh
                - F.softplus(-fh)
                + theta * log_1_minus_p
                + x * log_p
                + (x + theta).lgamma()
                - theta.lgamma()
                - (x + 1).lgamma()
            )
            log_prob = torch.where(torch.le(x, 1e-8), zero_ll, non_zero_ll)
        else:
            raise NotImplementedError

        return log_prob.sum(dim=1)


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
    batch = list(map(lambda b: {'x': b[0], 's': b[1], 'c': b[2]}, batch))
    batch = merge_with(torch.vstack, batch)
    batch['x'] = batch['x'].float()
    batch['s'] = F.one_hot(batch['s'], num_classes=num_batches).squeeze(1).float()
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

    embeddings = model.get_latent_representation()
    np.save(f'{checkpoint_dir}/embeddings.npy', embeddings)
    np.save(f'{checkpoint_dir}/mde.npy', scvi.model.utils.mde(embeddings))

    Path(f'{checkpoint_dir}/DONE').touch()

    torch.cuda.empty_cache()


def fit_model(
        batch_size: int,
        dataset: str,
        learning_rate: float,
        model_config: dict,
        checkpoint_dir: str,
        scvi_dir: str,
        seed: int | None
):
    if seed is not None:
        torch.manual_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    scvi_dataset = load_dataset(dataset)
    i_train = np.load(f'{scvi_dir}/train_indices.npy')
    i_valid = np.load(f'{scvi_dir}/valid_indices.npy')

    cell_types = np.unique(scvi_dataset.obs['cell_type'], return_inverse=True)[1]
    n_batches = len(np.unique(scvi_dataset.obs['batch']))
    n_cells = scvi_dataset.layers['counts'].shape[0]
    n_genes = scvi_dataset.layers['counts'].shape[1]

    ds_train = TensorDataset(
        torch.tensor(scvi_dataset.layers['counts'][i_train]),
        torch.LongTensor(scvi_dataset.obs['batch'].values[i_train]),
        torch.LongTensor(cell_types[i_train]),
    )
    ds_valid = TensorDataset(
        torch.tensor(scvi_dataset.layers['counts'][i_valid]),
        torch.LongTensor(scvi_dataset.obs['batch'].values[i_valid]),
        torch.LongTensor(cell_types[i_valid]),
    )

    dl_kwargs = {
        'batch_size': batch_size,
        'collate_fn': curry(collate, num_batches=n_batches),
        'generator': g,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': False,
        'worker_init_fn': seed_worker if seed is not None else None,
    }
    dl_train = DataLoader(ds_train, drop_last=True, shuffle=True, **dl_kwargs)
    dl_valid = DataLoader(ds_valid, drop_last=False, shuffle=False, **dl_kwargs)

    encoder = Encoder(
        dim_in=n_genes + n_batches,
        dim_out=DATASET_CONFIG[dataset]['n_latent'],
        **DATASET_CONFIG[dataset],
    )
    decoder = Decoder(
        dim_in=DATASET_CONFIG[dataset]['n_latent'] + n_batches,
        dim_out=n_genes,
        **DATASET_CONFIG[dataset]
    )
    u_initial = next(iter(dl_train))
    u_initial = torch.concat([
        u_initial['x'][:TRUNCATION_LEVEL].log1p(),
        u_initial['s'][:TRUNCATION_LEVEL]
    ], dim=1)
    model = SingleCellVampPrior(
        n_cells=n_cells,
        n_genes=n_genes,
        n_batches=n_batches,
        n_latent=DATASET_CONFIG[dataset]['n_latent'],
        encoder=encoder,
        decoder=decoder,
        u_initial=u_initial,
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

    embeddings = trainer.predict(model, DataLoader(ConcatDataset([ds_train, ds_valid]), shuffle=False, **dl_kwargs))
    np.save(f'{checkpoint_dir}/embeddings.npy', embeddings)
    np.save(f'{checkpoint_dir}/mde.npy', scvi.model.utils.mde(embeddings))

    Path(f'{checkpoint_dir}/DONE').touch()

    torch.cuda.empty_cache()


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
        results_dir = f'results/scRNA/{args.dataset}/{args.seed}/trial_{trial}'
        os.makedirs(results_dir, exist_ok=True)

        scvi_checkpoint_path = f'experiments/scRNA/{args.dataset}/{args.seed}/trial_{trial}/scVI'
        os.makedirs(scvi_checkpoint_path, exist_ok=True)
        if not os.path.exists(f'{scvi_checkpoint_path}/DONE') or args.replace_results:
            fit_scvi(args.batch_size, args.dataset, args.learning_rate, scvi_checkpoint_path, trial_seed)
        plot_mdes(args.dataset, scvi_checkpoint_path, results_dir, 'scVI.pdf')

        for config in TEST_CASES:
            checkpoint_path = f'experiments/scRNA/{args.dataset}/{args.seed}/trial_{trial}/{str(config)}'
            os.makedirs(checkpoint_path, exist_ok=True)
            if not os.path.exists(f'{checkpoint_path}/DONE') or args.replace_results:
                fit_model(
                    args.batch_size,
                    args.dataset,
                    args.learning_rate,
                    config,
                    checkpoint_path,
                    scvi_checkpoint_path,
                    trial_seed
                )
        #     plot_mdes(args.dataset, scvi_checkpoint_path, results_dir, f'{str(config)}.pdf')

    # results_dir = f'results/scRNA/{args.dataset}/{args.seed}'
    # os.makedirs(results_dir, exist_ok=True)
    # build_table(results, results_dir)
