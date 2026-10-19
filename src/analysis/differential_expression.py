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
from torch.utils.data import DataLoader

from src.data.scRNA import scRNADataModule
from src.distributions import concat_distributions
from src.experiments.utils import checkpoint_path
from src.model_wrapper import ModelWrapper


def _load_counts(datamodule, index, device):
    """Dense float tensor of the int counts the model saw, for the given cell index."""
    X = datamodule.counts[index].tocsr()
    return torch.sparse_csr_tensor(
        torch.tensor(X.indptr, dtype=torch.int64),
        torch.tensor(X.indices, dtype=torch.int64),
        torch.tensor(X.data, dtype=torch.float32),
        size=X.shape,
        device=device,
    ).to_dense()


def _pseudobulk(y, y_hat, obs, cols, min_cells):
    """Sum counts per (condition, batch, cell_type) group, normalize, log1p(CP10k).

    y_hat may be None (observed-only split). Returns DataFrames indexed by group,
    restricted to groups with >= min_cells cells.
    """
    grouped = obs.groupby(cols, observed=True, sort=False)
    codes = torch.tensor(
        grouped.ngroup().to_numpy(), dtype=torch.int64, device=y.device
    )
    group_index = grouped.size().index
    n_groups = len(group_index)
    counts = torch.bincount(codes, minlength=n_groups)

    sum_obs = torch.zeros(n_groups, y.shape[1], device=y.device, dtype=y.dtype)
    sum_obs.index_add_(0, codes, y)

    eps = 1e-8

    def lognorm(s):
        return (s / s.sum(1, keepdim=True).clamp_min(eps) * 1e4).log1p()

    large = (counts >= min_cells).cpu().numpy()
    out = {
        "log_obs": pd.DataFrame(lognorm(sum_obs).numpy(force=True), index=group_index)[
            large
        ],
        "raw_obs": pd.DataFrame(sum_obs.numpy(force=True), index=group_index)[large],
    }
    if y_hat is not None:
        sum_pred = torch.zeros_like(sum_obs)
        sum_pred.index_add_(0, codes, y_hat)
        out["log_pred"] = pd.DataFrame(
            lognorm(sum_pred).numpy(force=True), index=group_index
        )[large]
    print(
        f"  dropping {int((~large).sum())}/{n_groups} groups with < {min_cells} cells"
    )
    return out


def _topk_overlap(de_a, de_b, valid, ks):
    """Mean top-k overlap on |DE| per (batch, cell_type) pair, plus chance level."""
    res = {}
    for k in ks:
        overlaps, chance = [], []
        for i in range(de_a.shape[0]):
            v = np.where(valid[i])[0]
            if len(v) < 2 * k:
                continue
            top_a = set(v[np.argsort(-np.abs(de_a[i, v]))[:k]])
            top_b = set(v[np.argsort(-np.abs(de_b[i, v]))[:k]])
            overlaps.append(len(top_a & top_b) / k)
            chance.append(k / len(v))
        res[k] = (
            (float(np.mean(overlaps)), float(np.mean(chance)), len(overlaps))
            if overlaps
            else (float("nan"), float("nan"), 0)
        )
    return res


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
    min_cells: int = 30,  # min cells per group, required in BOTH train and test
    min_gene_counts: int = 30,  # min observed pseudobulk counts per gene, in BOTH groups and BOTH splits
    top_ks: tuple[int, ...] = (50, 100, 200),
):
    cols = [condition_column, "batch", "cell_type"]
    all_obs = datamodule.adata.obs[cols].reset_index(drop=True)  # read adata once

    # ---- model predictions on TRAINING cells (unshuffled loader, train_index order) ----
    train_loader = DataLoader(
        datamodule.data["train"],
        shuffle=False,
        **datamodule._dl_kwargs(split="val"),
    )
    predictions = trainer.predict(
        model, train_loader, ckpt_path=checkpoint_path(checkpoint_callback)
    )
    predictions = merge_with(concat_distributions, predictions)
    y_hat_tr = predictions["px"].mean  # (1 - sigmoid(rho)) * mu
    device = y_hat_tr.device

    # ---- training split: observed counts + predictions ----
    train_index = np.asarray(datamodule.train_index)
    y_tr = _load_counts(datamodule, train_index, device)
    assert y_hat_tr.shape == y_tr.shape, (
        f"prediction shape {tuple(y_hat_tr.shape)} != train data shape "
        f"{tuple(y_tr.shape)}; check that the loader is unshuffled and matches train_index"
    )
    obs_tr = all_obs.iloc[train_index].reset_index(drop=True)

    # ---- test split: observed counts only ----
    test_index = np.asarray(datamodule.test_index)
    y_te = _load_counts(datamodule, test_index, device)
    obs_te = all_obs.iloc[test_index].reset_index(drop=True)

    # drop zero-count cells in each split
    valid_tr = (y_tr.sum(1) > 0).cpu()
    if (~valid_tr).any():
        print(f"Dropping {int((~valid_tr).sum())} zero-count train cells")
        k_ = valid_tr.to(device)
        y_tr, y_hat_tr = y_tr[k_], y_hat_tr[k_]
        obs_tr = obs_tr.loc[valid_tr.numpy()].reset_index(drop=True)
    valid_te = (y_te.sum(1) > 0).cpu()
    if (~valid_te).any():
        print(f"Dropping {int((~valid_te).sum())} zero-count test cells")
        y_te = y_te[valid_te.to(device)]
        obs_te = obs_te.loc[valid_te.numpy()].reset_index(drop=True)

    # diagnostics: library agreement and row-order check (train side)
    l_obs, l_pred = y_tr.sum(1), y_hat_tr.sum(1)
    r_lib = torch.corrcoef(torch.stack([l_obs, l_pred]))[0, 1].item()
    print(
        f"train: mean observed library {l_obs.mean().item():.1f} | "
        f"mean predicted ZI total {l_pred.mean().item():.1f} | "
        f"per-cell library corr {r_lib:.3f}"
    )
    assert (
        r_lib > 0.8
    ), "Row order likely mismatched between predictions and train_index"

    # ---- pseudobulk each split ----
    print("train split:")
    pb_tr = _pseudobulk(y_tr, y_hat_tr, obs_tr, cols, min_cells)
    print("test split:")
    pb_te = _pseudobulk(y_te, None, obs_te, cols, min_cells)

    # ---- align (batch, cell_type) pairs across conditions AND splits ----
    sel = {("tr", g): {k: v.loc[g] for k, v in pb_tr.items()} for g in (group1, group2)}
    sel.update(
        {("te", g): {k: v.loc[g] for k, v in pb_te.items()} for g in (group1, group2)}
    )

    common = sel[("tr", group1)]["log_obs"].index
    for key in sel:
        common = common.intersection(sel[key]["log_obs"].index)
    assert len(common) > 0, (
        "No (batch, cell_type) pairs pass min_cells in both conditions and both splits; "
        "lower min_cells or check what `batch` encodes"
    )
    for key in sel:
        sel[key] = {k: v.loc[common] for k, v in sel[key].items()}

    # ---- gene filter: observed counts >= threshold in both groups, both splits ----
    # (observed only, so identical across priors)
    gene_ok = np.ones(sel[("tr", group1)]["raw_obs"].shape, dtype=bool)
    for key in sel:
        gene_ok &= sel[key]["raw_obs"].values >= min_gene_counts
    print(
        f"{len(common)} (batch, cell_type) pairs; gene filter keeps "
        f"{int(gene_ok.sum())}/{gene_ok.size} (pair, gene) entries"
    )

    # ---- contrasts (pairs x genes) ----
    de_pred_tr = (
        sel[("tr", group1)]["log_pred"].values - sel[("tr", group2)]["log_pred"].values
    )
    de_obs_tr = (
        sel[("tr", group1)]["log_obs"].values - sel[("tr", group2)]["log_obs"].values
    )
    de_obs_te = (
        sel[("te", group1)]["log_obs"].values - sel[("te", group2)]["log_obs"].values
    )

    ok = (
        gene_ok
        & np.isfinite(de_pred_tr)
        & np.isfinite(de_obs_tr)
        & np.isfinite(de_obs_te)
    )
    x, y_ = de_obs_te[ok], de_pred_tr[ok]
    x_ref, y_ref = de_obs_te[ok], de_obs_tr[ok]
    print(f"{group1} vs {group2}: {x.shape[0]} DE values")
    assert x.size > 1, "No DE values left after filtering; relax the thresholds"

    r_p, r_s = pearsonr(x, y_)[0], spearmanr(x, y_)[0]
    ref_p, ref_s = pearsonr(x_ref, y_ref)[0], spearmanr(x_ref, y_ref)[0]
    print(f"model(train) vs obs(test): Pearson {r_p:.3f}, Spearman {r_s:.3f}")
    print(f"obs(train)   vs obs(test): Pearson {ref_p:.3f}, Spearman {ref_s:.3f}")

    # ---- top-k overlap on |DE|, per (batch, cell_type) pair ----
    ov = _topk_overlap(de_pred_tr, de_obs_te, ok, top_ks)
    ov_ref = _topk_overlap(de_obs_tr, de_obs_te, ok, top_ks)
    for k in top_ks:
        print(
            f"top-{k} |DE| overlap: model {ov[k][0]:.3f} | obs-train {ov_ref[k][0]:.3f} "
            f"| chance {ov[k][1]:.3f} | pairs used {ov[k][2]}"
        )

    suffix = f"_{tag}" if tag else ""
    np.save(Path(save_path) / f"de_pred_train{suffix}.npy", y_)
    np.save(Path(save_path) / f"de_obs_test{suffix}.npy", x)
    np.save(Path(save_path) / f"de_obs_train{suffix}.npy", y_ref)
    # full pair x gene arrays for later per-cell-type analysis / bootstraps
    np.savez(
        Path(save_path) / f"de_arrays{suffix}.npz",
        de_pred_train=de_pred_tr,
        de_obs_train=de_obs_tr,
        de_obs_test=de_obs_te,
        ok=ok,
        pairs=np.array([str(p) for p in common]),
    )

    # ---- plot ----
    plt.figure(figsize=(8, 8))
    sns.regplot(x=x, y=y_, scatter_kws={"rasterized": True, "alpha": 0.5})
    plt.xlabel("Observed DE (test cells)")
    plt.ylabel("Predicted DE (model, train cells)")
    plt.title(f"Differential Expression: {group1} vs {group2}")
    overlap_lines = "\n".join(
        f"top-{k} |DE| overlap = {ov[k][0]:.3f} (chance {ov[k][1]:.3f})" for k in top_ks
    )
    plt.text(
        0.05,
        0.95,
        f"Pearson r = {r_p:.3f}\nSpearman r = {r_s:.3f}\n{overlap_lines}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )
    plt.savefig(Path(save_path) / f"de_scatter{suffix}.pdf")
    plt.close()
