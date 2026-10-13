from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from scipy.stats import pearsonr, spearmanr
from toolz import merge_with

from src.data.scRNA import scRNADataModule
from src.distributions import concat_distributions
from src.experiments.utils import checkpoint_path
from src.model_wrapper import ModelWrapper


def differential_expression(
    datamodule: scRNADataModule,
    model: ModelWrapper,
    trainer: Trainer,
    save_path: str,
    *,
    checkpoint_callback: ModelCheckpoint,
    condition_column: str,
    group1: str,
    group2: str,
    tag: str = "",
):
    # get model predictions
    predictions = trainer.predict(
        model,
        datamodule.test_dataloader(),
        ckpt_path=checkpoint_path(checkpoint_callback),
    )
    predictions = merge_with(concat_distributions, predictions)

    # observed counts as a dense tensor on the same device as the predictions
    y_hat = predictions["px"].mu
    X = datamodule.adata.X[datamodule.test_index]
    y = torch.sparse_csr_tensor(
        torch.tensor(X.indptr, dtype=torch.int64),
        torch.tensor(X.indices, dtype=torch.int64),
        torch.tensor(X.data, dtype=torch.float32),
        size=X.shape,
        device=y_hat.device,
    ).to_dense()

    # sanity check: predictions must line up row-for-row with adata
    assert y_hat.shape == y.shape, (
        f"prediction shape {tuple(y_hat.shape)} != data shape {tuple(y.shape)}; "
        "check that the predict dataloader does not shuffle or drop cells"
    )

    obs = (
        datamodule.adata.obs[[condition_column, "batch", "cell_type"]]
        .reset_index(drop=True)
        .iloc[datamodule.test_index]
    )

    # drop zero-count cells (library size 0 would produce NaNs)
    l = y.sum(dim=1, keepdim=True)
    valid = (l.squeeze(1) > 0).cpu()
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(f"Dropping {n_dropped} cells with zero total counts")
        valid_np = valid.numpy()
        keep = valid.to(y.device)
        y, y_hat, l = y[keep], y_hat[keep], l[keep]
        obs = obs.loc[valid_np].reset_index(drop=True)

    # normalize predictions and observed counts: log1p(CP10k)
    y = (y / l * 10_000).log1p()
    y_hat = (y_hat / l * 10_000).log1p()

    # compute mean expression for each (condition, batch, cell_type) group
    cols = [condition_column, "batch", "cell_type"]
    grouped = obs.groupby(cols, observed=True, sort=False)
    codes = torch.tensor(
        grouped.ngroup().to_numpy(), dtype=torch.int64, device=y.device
    )
    group_index = grouped.size().index  # MultiIndex, same order as ngroup codes
    n_groups = len(group_index)

    onehot = torch.zeros(y.shape[0], n_groups, device=y.device, dtype=y.dtype)
    onehot[torch.arange(y.shape[0], device=y.device), codes] = 1.0
    counts = onehot.sum(dim=0, keepdim=True).T  # (n_groups, 1)

    df_mean_obs = pd.DataFrame(
        ((onehot.T @ y) / counts).numpy(force=True), index=group_index
    )
    df_mean_pred = pd.DataFrame(
        ((onehot.T @ y_hat) / counts).numpy(force=True), index=group_index
    )

    # compute conditional differential expression between the two groups
    de_obs = df_mean_obs.loc[group1] - df_mean_obs.loc[group2]
    de_pred = df_mean_pred.loc[group1] - df_mean_pred.loc[group2]

    de_obs = de_obs.values.reshape(-1)
    de_pred = de_pred.values.reshape(-1)

    # drop NaNs jointly so observed and predicted stay aligned
    mask = ~(np.isnan(de_obs) | np.isnan(de_pred))
    de_obs, de_pred = de_obs[mask], de_pred[mask]
    print(f"{group1} vs {group2}: {de_obs.shape[0]} DE values")
    assert de_obs.size > 1, (
        "No overlapping (batch, cell_type) entries between groups; "
        "check what `batch` encodes"
    )

    np.save(Path(save_path) / f"de_obs_{tag}.npy", de_obs)
    np.save(Path(save_path) / f"de_pred_{tag}.npy", de_pred)

    # plot all conditional differential expression
    plt.figure(figsize=(8, 8))
    sns.regplot(x=de_obs, y=de_pred, scatter_kws={"rasterized": True, alpha=0.5})
    plt.xlabel("Observed DE")
    plt.ylabel("Predicted DE")
    plt.title(f"Differential Expression: {group1} vs {group2}")
    plt.text(
        0.05,
        0.95,
        f"Pearson r = {pearsonr(de_obs, de_pred)[0]:.3f} \n"
        f"Spearman r = {spearmanr(de_obs, de_pred)[0]:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )
    plt.savefig(Path(save_path) / f"de_scatter_{tag}.pdf")
    plt.close()
