from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd
import pooch
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from anndata.io import read_elem


def _read_h5(f: h5py.Group, key: str) -> Any:
    return read_elem(cast(Any, f[key]))


def scread(
    root: str,
    filename: str,
    backup_url: str | None = None,
    counts_layer: str = "counts",
) -> AnnData:
    path = Path(root) / filename

    if not path.is_file():
        if backup_url is None:
            raise FileNotFoundError(f"{path} not found and no backup_url provided")
        path.parent.mkdir(parents=True, exist_ok=True)
        pooch.retrieve(
            url=backup_url, known_hash=None, fname=filename, path=str(path.parent)
        )

    with h5py.File(path, "r") as f:
        obs = _read_h5(f, "obs")
        var = _read_h5(f, "var")
        counts = _read_h5(f, f"layers/{counts_layer}")
        # scIBMetrics uses the pre-integration PCA as the baseline for the PCR metric
        obsm = {"X_pca": _read_h5(f, "obsm/X_pca")} if "obsm/X_pca" in f else None

    return AnnData(obs=obs, var=var, layers={counts_layer: counts}, obsm=obsm)


def rename_adata_obs(adata: AnnData, columns: dict[str, str]) -> AnnData:
    if isinstance(adata.obs, pd.DataFrame):
        adata.obs = adata.obs.rename(columns=columns)
    else:
        raise ValueError("adata.obs is not a pandas DataFrame")
    return adata


def constant_batch(adata: AnnData) -> AnnData:
    adata.obs["batch"] = 0
    return adata


def use_counts(adata: AnnData) -> AnnData:
    adata.X = adata.layers["counts"]
    return adata


def select_hvg_by_score(
    adata: AnnData, n_top_genes: int = 2000, key: str = "hvg_score"
) -> AnnData:
    idx = adata.var[key].to_numpy().argsort()[::-1][:n_top_genes]
    return adata[:, idx].copy()


def compute_preintegration_pca(counts: sp.spmatrix, n_comps: int = 50) -> np.ndarray:
    """log-CP10k → PCA (50 comps by default). Matches OpenProblems' pre-integration baseline."""
    counts_dense = np.asarray(counts.todense(), dtype=np.float64)
    log_cp10k = np.log1p(
        counts_dense / counts_dense.sum(axis=1, keepdims=True) * 10_000
    )
    X_pca = sc.pp.pca(
        log_cp10k,
        n_comps=min(n_comps, log_cp10k.shape[1] - 1, log_cp10k.shape[0] - 1),
        svd_solver="arpack",
    )
    if not isinstance(X_pca, np.ndarray):
        raise ValueError("X_pca is not a numpy array")
    return X_pca.astype(np.float32, copy=False)
