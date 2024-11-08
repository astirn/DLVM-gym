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
from dataclasses import dataclass
from models import SingleCellVampPrior
from torch.utils.data import DataLoader, TensorDataset
from utils import get_trainer, seed_worker
from pathlib import Path
from scib_metrics.benchmark import Benchmarker
from typing import Any, Dict, Union
from utils import process_results


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
PATIENCE = 10
TRUNCATION_LEVEL = 100


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

    def forward(self, batch: dict[str, torch.Tensor]) -> dist.MultivariateNormal:
        xs = torch.concat([batch['x'], batch['s']], dim=1)
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

    def forward(self, z: torch.Tensor, batch: dict[str, torch.Tensor], eps=1e-8):
        x, s = batch['x'], batch['s']
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
    batch['x'] = batch['x'].float().log1p()
    batch['s'] = F.one_hot(batch['s'], num_classes=num_batches).squeeze(1).float()
    batch['c'] = batch['c'].squeeze()
    return batch


def get_dataloaders(
        batch_size: int,
        dataset: anndata.AnnData,
        seed: int | None,
        split_dir: str = None
):
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    cell_types = np.unique(dataset.obs['cell_type'], return_inverse=True)[1]

    dl_kwargs = {
        'batch_size': batch_size,
        'collate_fn': curry(collate, num_batches=len(np.unique(dataset.obs['batch']))),
        'generator': g,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': False,
        'worker_init_fn': seed_worker if seed is not None else None,
    }

    if split_dir is None:
        ds = TensorDataset(
            torch.tensor(dataset.layers['counts']),
            torch.LongTensor(dataset.obs['batch'].values),
            torch.LongTensor(cell_types),
        )

        return DataLoader(ds, drop_last=False, shuffle=False, **dl_kwargs)

    else:
        i_train = np.load(f'{split_dir}/train_indices.npy')
        i_valid = np.load(f'{split_dir}/valid_indices.npy')

        ds_train = TensorDataset(
            torch.tensor(dataset.layers['counts'][i_train]),
            torch.LongTensor(dataset.obs['batch'].values[i_train]),
            torch.LongTensor(cell_types[i_train]),
        )
        ds_valid = TensorDataset(
            torch.tensor(dataset.layers['counts'][i_valid]),
            torch.LongTensor(dataset.obs['batch'].values[i_valid]),
            torch.LongTensor(cell_types[i_valid]),
        )

        dl_train = DataLoader(ds_train, drop_last=True, shuffle=True, **dl_kwargs)
        dl_valid = DataLoader(ds_valid, drop_last=False, shuffle=False, **dl_kwargs)

        return dl_train, dl_valid


def fit_scvi(
        batch_size: int,
        dataset: anndata.AnnData,
        dataset_config: dict,
        learning_rate: float,
        checkpoint_dir: str,
        seed: int | None
):
    if seed is not None:
        torch.manual_seed(seed)

    scvi.model.SCVI.setup_anndata(dataset, layer='counts', batch_key='batch')
    model = scvi.model.SCVI(
        adata=dataset,
        deeply_inject_covariates=False,
        log_variational=True,
        **dataset_config
    )

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
    np.save(f'{checkpoint_dir}/cell_embeddings.npy', model.get_latent_representation())

    Path(f'{checkpoint_dir}/DONE').touch()

    torch.cuda.empty_cache()


def fit_model(
        batch_size: int,
        dataset: anndata.AnnData,
        dataset_config: dict,
        learning_rate: float,
        model_config: dict,
        checkpoint_dir: str,
        seed: int | None,
        split_dir: str
):
    if seed is not None:
        torch.manual_seed(seed)

    n_batches = len(np.unique(dataset.obs['batch']))
    n_cells = dataset.layers['counts'].shape[0]
    n_genes = dataset.layers['counts'].shape[1]

    dl_train, dl_valid = get_dataloaders(batch_size, dataset, seed, split_dir)

    encoder = Encoder(
        dim_in=n_genes + n_batches,
        dim_out=dataset_config['n_latent'],
        **dataset_config,
    )
    decoder = Decoder(
        dim_in=dataset_config['n_latent'] + n_batches,
        dim_out=n_genes,
        **dataset_config
    )
    u_initial = next(iter(dl_train))
    u_initial = torch.concat([
        u_initial['x'][:TRUNCATION_LEVEL].log1p(),
        u_initial['s'][:TRUNCATION_LEVEL]
    ], dim=1)
    model = SingleCellVampPrior(
        dim_z=dataset_config['n_latent'],
        num_data=n_cells,
        num_genes=n_genes,
        encoder=encoder,
        decoder=decoder,
        u_initial=u_initial,
        truncation_level=TRUNCATION_LEVEL,
        learning_rate=learning_rate,
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


def test_model(
        batch_size: int,
        dataset: anndata.AnnData,
        checkpoint_dir: str,
        seed: int | None,
) -> (np.ndarray, np.ndarray):
    if seed is not None:
        torch.manual_seed(seed)

    trainer = get_trainer(checkpoint_metrics=[], checkpoint_dir=None, patience=0)

    predictions = trainer.predict(
        SingleCellVampPrior.load_from_checkpoint(f"{checkpoint_dir}/best_val_elbo.ckpt"),
        get_dataloaders(batch_size, dataset, seed)
    )
    predictions = merge_with(lambda x: torch.vstack(x).cpu().numpy(), predictions)

    torch.cuda.empty_cache()

    return predictions.values()


def plot_mde(
        cell_embeddings: np.ndarray,
        cluster_probabilities: np.ndarray | None,
        dataset: anndata.AnnData,
        fig_path: str,
        fig_name: str,
        seed: int | None
):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))

    mde = scvi.model.utils.mde(cell_embeddings, seed=seed)

    if cluster_probabilities is not None:
        cluster_assignments = np.argmax(cluster_probabilities, axis=1)
        _, cluster_assignments = np.unique(cluster_assignments, return_inverse=True)
    else:
        cluster_assignments = np.zeros_like(dataset.obs['cell_type'])

    df = pd.DataFrame(data={
        'x': mde[:, 0],
        'y': mde[:, 1],
        'Batch': dataset.obs['batch'] + 1,
        'Cell type': dataset.obs['cell_type'],
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
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, markerscale=5)

    axes[0].set_title('Batch correction')
    axes[1].set_title('Bio conservation')
    axes[2].set_title('Model\'s Cluster Assignments')

    sns.move_legend(axes[0], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
    sns.move_legend(axes[1], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
    sns.move_legend(axes[2], loc='upper center', bbox_to_anchor=(0.5, 0), ncol=4)

    plt.tight_layout()
    fig.savefig(f'{fig_path}/{fig_name}', dpi=200)


@dataclass(frozen=True)
class NoBatchCorrection:
    """
    A class to skip batch correction metrics that won't work if only a single batch exists
    """
    silhouette_batch: Union[bool, Dict[str, Any]] = False
    ilisi_knn: Union[bool, Dict[str, Any]] = False
    kbet_per_label: Union[bool, Dict[str, Any]] = False
    graph_connectivity: Union[bool, Dict[str, Any]] = False
    pcr_comparison: Union[bool, Dict[str, Any]] = True


def config_to_index(configuration: dict):
    if configuration.get('dirichlet_prior'):
        cluster_proportions = 'Dirichlet prior'
    elif configuration.get('dirichlet_process'):
        cluster_proportions = 'DP prior'
    else:
        cluster_proportions = '--'

    if configuration.get('decoupled_variance'):
        cluster_widths = 'Decoupled'
    elif configuration.get('wishart_prior'):
        cluster_widths = 'Wishart prior'
    else:
        cluster_widths = '--'

    return {
        'Cluster Proportions': cluster_proportions,
        'Cluster Widths': cluster_widths,
    }


def table_index(df: pd.DataFrame, dataset_name: str) -> pd.MultiIndex:
    index = [
        {
            'Dataset': dataset_name,
            'Embedding Prior': '$\\N(0,I)$' if i == 'scVI' else "VP",
            **(config_to_index({}) if i == 'scVI' else config_to_index(eval(i)))
        }
        for i in df.index.values
    ]
    return pd.MultiIndex.from_frame(pd.DataFrame(index))


def integration_performance(dataset: anndata.AnnData, dataset_name: str) -> pd.DataFrame:
    single_batch = dataset.obs['batch'].nunique() == 1

    bm = Benchmarker(
        dataset,
        batch_key='batch',
        label_key='cell_type',
        embedding_obsm_keys=['scVI'] + [str(t) for t in TEST_CASES],
        batch_correction_metrics=NoBatchCorrection() if single_batch else None,
        n_jobs=-1,
    )
    bm.benchmark()

    df = bm.get_results(min_max_scale=False).copy()
    df.columns = pd.MultiIndex.from_arrays([df.loc['Metric Type'], df.columns], names=[None, None])
    df = df.loc[~df.index.isin(['Metric Type'])].astype(float)
    if single_batch:
        df = df[[col for col in df.columns if 'Bio conservation' in col]]
    df.index = table_index(df, dataset_name)

    return df


def build_tables(integration_results: list[pd.DataFrame], save_path: str):
    integration_table = pd.concat(integration_results)
    integration_table = process_results(integration_table)
    integration_table.to_pickle(f'{save_path}/integration_results.pkl')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--dataset', type=str, default='cortex')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--replace_results', action='store_true')
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')

    data = load_dataset(args.dataset)

    model_results = []
    sc_results = []
    for trial in range(1, args.num_trials + 1):
        trial_seed = zlib.crc32(f'{args.seed}{trial}'.encode())
        results_dir = f'results/scRNA/{args.seed}/{args.dataset}/trial_{trial}'
        os.makedirs(results_dir, exist_ok=True)

        scvi_dir = f'experiments/scRNA/{args.seed}/{args.dataset}/trial_{trial}/scVI'
        os.makedirs(scvi_dir, exist_ok=True)
        if not os.path.exists(f'{scvi_dir}/DONE') or args.replace_results:
            fit_scvi(
                args.batch_size,
                data,
                DATASET_CONFIG[args.dataset],
                args.learning_rate,
                scvi_dir,
                trial_seed
            )
        embeddings = np.load(f'{scvi_dir}/cell_embeddings.npy')
        data.obsm['scVI'] = embeddings
        plot_mde(
            embeddings,
            None,
            data,
            results_dir,
            'scVI.pdf',
            trial_seed
        )

        for config in TEST_CASES:
            checkpoint_path = f'experiments/scRNA/{args.seed}/{args.dataset}/trial_{trial}/{str(config)}'
            os.makedirs(checkpoint_path, exist_ok=True)
            if not os.path.exists(f'{checkpoint_path}/DONE') or args.replace_results:
                fit_model(
                    args.batch_size,
                    data,
                    DATASET_CONFIG[args.dataset],
                    args.learning_rate,
                    config,
                    checkpoint_path,
                    trial_seed,
                    scvi_dir,
                )
            embeddings, cluster_probs = test_model(
                args.batch_size,
                data,
                checkpoint_path,
                trial_seed,
            )
            data.obsm[str(config)] = embeddings
            plot_mde(
                embeddings,
                cluster_probs,
                data,
                results_dir,
                f'{str(config)}.pdf',
                trial_seed
            )

        sc_results.append(integration_performance(data, args.dataset))

    results_dir = f'results/scRNA/{args.seed}/{args.dataset}'
    build_tables(sc_results, results_dir)
