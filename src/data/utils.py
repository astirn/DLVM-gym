import pandas as pd
from anndata import AnnData


def rename_adata_obs(adata: AnnData, columns: dict[str, str]) -> AnnData:
    if isinstance(adata.obs, pd.DataFrame):
        adata.obs = adata.obs.rename(columns=columns)
    else:
        raise ValueError("adata.obs is not a pandas DataFrame")
    return adata


def constant_batch(adata: AnnData) -> AnnData:
    adata.obs["batch"] = 0
    return adata
