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
    min_cells: int = 30,  # min cells per (condition, batch, cell_type) group
    min_gene_counts: int = 30,  # min observed pseudobulk counts for a gene in BOTH groups
):
    # get model predictions on the test set
    predictions = trainer.predict(
        model,
        datamodule.test_dataloader(),
        ckpt_path=checkpoint_path(checkpoint_callback),
    )
    predictions = merge_with(concat_distributions, predictions)
    y_hat = predictions["px"].mean

    # observed raw counts for the test cells only, as a dense tensor
    test_index = np.asarray(datamodule.test_index)
    X = datamodule.counts[
        test_index
    ].tocsr()  # the int16/int32 matrix the model trained on
    y = torch.sparse_csr_tensor(
        torch.tensor(X.indptr, dtype=torch.int64),
        torch.tensor(X.indices, dtype=torch.int64),
        torch.tensor(X.data, dtype=torch.float32),  # cast ints to float
        size=X.shape,
        device=y_hat.device,
    ).to_dense()

    assert y_hat.shape == y.shape, (
        f"prediction shape {tuple(y_hat.shape)} != data shape {tuple(y.shape)}; "
        "check that the predict dataloader does not shuffle or drop cells and "
        "that its order matches datamodule.test_index"
    )

    obs = (
        datamodule.adata.obs[[condition_column, "batch", "cell_type"]]
        .reset_index(drop=True)
        .iloc[test_index]
        .reset_index(drop=True)
    )

    # drop cells with zero observed counts
    valid = (y.sum(dim=1) > 0).cpu()
    n_dropped = int((~valid).sum())
    if n_dropped > 0:
        print(f"Dropping {n_dropped} cells with zero total counts")
        keep = valid.to(y.device)
        y, y_hat = y[keep], y_hat[keep]
        obs = obs.loc[valid.numpy()].reset_index(drop=True)

    # diagnostic: predicted total should be roughly comparable to observed total
    print(
        f"mean observed library: {y.sum(1).mean().item():.1f} | "
        f"mean predicted ZI total: {y_hat.sum(1).mean().item():.1f}"
    )

    # group cells by (condition, batch, cell_type)
    cols = [condition_column, "batch", "cell_type"]
    grouped = obs.groupby(cols, observed=True, sort=False)
    codes = torch.tensor(
        grouped.ngroup().to_numpy(), dtype=torch.int64, device=y.device
    )
    group_index = grouped.size().index
    n_groups = len(group_index)
    counts = torch.bincount(codes, minlength=n_groups)

    # pseudobulk: sum raw observed counts and ZI expected counts within each group
    sum_obs = torch.zeros(n_groups, y.shape[1], device=y.device, dtype=y.dtype)
    sum_obs.index_add_(0, codes, y)
    sum_pred = torch.zeros_like(sum_obs)
    sum_pred.index_add_(0, codes, y_hat)

    # normalize each pseudobulk by its own total over ALL genes, then log1p(CP10k)
    eps = 1e-8
    log_obs = (sum_obs / sum_obs.sum(1, keepdim=True).clamp_min(eps) * 1e4).log1p()
    log_pred = (sum_pred / sum_pred.sum(1, keepdim=True).clamp_min(eps) * 1e4).log1p()

    # drop small groups (defined on observed cell counts, identical across priors)
    large = (counts >= min_cells).cpu().numpy()
    if min_cells > 0:
        print(
            f"Dropping {int((~large).sum())}/{n_groups} groups with < {min_cells} cells"
        )

    df_log_obs = pd.DataFrame(log_obs.numpy(force=True), index=group_index)[large]
    df_log_pred = pd.DataFrame(log_pred.numpy(force=True), index=group_index)[large]
    df_raw_obs = pd.DataFrame(sum_obs.numpy(force=True), index=group_index)[large]

    # select each condition; the remaining index levels are (batch, cell_type)
    lo1, lo2 = df_log_obs.loc[group1], df_log_obs.loc[group2]
    lp1, lp2 = df_log_pred.loc[group1], df_log_pred.loc[group2]
    r1, r2 = df_raw_obs.loc[group1], df_raw_obs.loc[group2]

    # keep only (batch, cell_type) pairs present in both conditions
    common = lo1.index.intersection(lo2.index)
    assert len(common) > 0, (
        "No overlapping (batch, cell_type) entries between groups after filtering; "
        "check what `batch` encodes and consider lowering min_cells"
    )
    lo1, lo2, lp1, lp2, r1, r2 = (d.loc[common] for d in (lo1, lo2, lp1, lp2, r1, r2))

    # per-contrast gene filter: observed pseudobulk counts >= threshold in BOTH groups
    # (observed counts only, so the filter is identical across priors)
    gene_ok = (r1.values >= min_gene_counts) & (r2.values >= min_gene_counts)
    print(
        f"{len(common)} (batch, cell_type) pairs; gene filter keeps "
        f"{int(gene_ok.sum())}/{gene_ok.size} (pair, gene) entries"
    )

    de_obs = (lo1.values - lo2.values)[gene_ok]
    de_pred = (lp1.values - lp2.values)[gene_ok]

    # drop NaNs jointly so observed and predicted stay aligned
    mask = ~(np.isnan(de_obs) | np.isnan(de_pred))
    de_obs, de_pred = de_obs[mask], de_pred[mask]
    print(f"{group1} vs {group2}: {de_obs.shape[0]} DE values")
    assert de_obs.size > 1, "No DE values left after filtering; relax the thresholds"

    np.save(Path(save_path) / f"de_obs_{tag}.npy", de_obs)
    np.save(Path(save_path) / f"de_pred_{tag}.npy", de_pred)

    plt.figure(figsize=(8, 8))
    sns.regplot(x=de_obs, y=de_pred, scatter_kws={"rasterized": True, "alpha": 0.5})
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
