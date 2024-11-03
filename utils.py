import numpy
import lightning
import lightning.pytorch.callbacks as callbacks
import random

import pandas as pd
import torch

from scipy.stats import ks_2samp


def get_trainer(
        early_stopping_metric: str = None,
        checkpoint_metrics: list[str] = None,
        checkpoint_dir: str = None,
        min_delta: float = 1.0,
        patience: int = None,
) -> lightning.Trainer:

    trainer_callbacks = [
        callbacks.ModelCheckpoint(
            auto_insert_metric_name=False,
            dirpath=checkpoint_dir,
            enable_version_counter=False,
            filename=f'best_{metric}',
            mode='max',
            monitor=metric,
            save_last=True,
            save_top_k=1,
        )
        for metric in checkpoint_metrics
    ] if checkpoint_metrics is not None else []
    if early_stopping_metric is not None:
        trainer_callbacks += [
            callbacks.EarlyStopping(
                min_delta=min_delta,
                monitor=early_stopping_metric,
                mode='max',
                patience=patience,
                verbose=True
            )
        ]

    return lightning.Trainer(
        callbacks=trainer_callbacks,
        devices=1,
        enable_checkpointing=checkpoint_dir is not None,
        logger=False,
        max_epochs=-1,
        num_sanity_val_steps=0,
    )


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def process_results(
        df: pd.DataFrame,
        threshold: float = 0.5,
        maximize: bool = True,
        pm_std: bool = False,
        no_test_cols: list = None
) -> pd.DataFrame:
    mean = df.groupby(level=df.index.names, sort=False).mean()
    std = df.groupby(level=df.index.names, sort=False).std()
    df.sort_index(inplace=True)
    for col in mean.columns:
        best = mean[col].idxmax() if maximize else mean[col].idxmin()
        for i in mean.index:
            if pm_std and mean.loc[i, col] > 1:
                mean.loc[i, col] = '{:.3g} $\\pm$ {:.2f}'.format(mean.loc[i, col], std.loc[i, col])
            elif pm_std:
                mean.loc[i, col] = '{:.3f} $\\pm$ {:.2f}'.format(mean.loc[i, col], std.loc[i, col])
            else:
                mean.loc[i, col] = '{:.4f}'.format(mean.loc[i, col])
            p = ks_2samp(df.loc[best, col], df.loc[i, col]).pvalue if i != best else None
            if (i == best or p >= threshold) and (no_test_cols is None or col not in no_test_cols):
                mean.loc[i, col] = '\\textbf{' + mean.loc[i, col] + '}'

    return mean
