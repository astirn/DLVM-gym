from pathlib import Path

import pandas as pd
import scanpy as sc
from anndata import AnnData


def scread(root: str, filename: str, backup_url: str) -> AnnData:
    path = Path(root) / filename
    return sc.read(path, backup_url=backup_url)


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
