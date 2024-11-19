import lightning
import lightning.pytorch.callbacks as callbacks
import numpy
import pandas as pd
import random
import scipy.stats as stats
import torch

from constants import TABLE_COLUMNS


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


def merge_results(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.rename(columns=TABLE_COLUMNS, inplace=True)
    df.fillna(False, inplace=True)

    table_columns = [col for col in TABLE_COLUMNS.values() if col in df.columns]
    df = df[table_columns]
    index = list(table_columns)
    index = index[:index.index('$\\log p(x)$')]
    df.set_index(index, inplace=True)

    if len(df[df.index.get_level_values('DP Prior')].index.get_level_values('DP VF').unique()) < 2:
        df = df.droplevel('DP VF')

    return df

def process_results(
        df: pd.DataFrame,
        min_cols: list = None,
        pm_std: bool = False,
        stat_test: str = 'wilcoxon',
        threshold: float = 0.05,
) -> pd.DataFrame:
    mean = df.groupby(level=df.index.names, sort=False).mean()
    std = df.groupby(level=df.index.names, sort=False).std()
    df.sort_index(inplace=True)
    min_cols = min_cols or []
    stat_test = getattr(stats, stat_test)
    for col in mean.columns:
        best = mean[col].idxmin() if col in min_cols else mean[col].idxmax()
        for i in mean.index:
            if pm_std and mean.loc[i, col] > 1:
                mean.loc[i, col] = '{:.3g} $\\pm$ {:.2f}'.format(mean.loc[i, col], std.loc[i, col])
            elif pm_std:
                mean.loc[i, col] = '{:.3f} $\\pm$ {:.2f}'.format(mean.loc[i, col], std.loc[i, col])
            else:
                mean.loc[i, col] = '{:.4f}'.format(mean.loc[i, col])

            if i == best:
                mean.loc[i, col] = '\\textbf{' + mean.loc[i, col] + '}'
            else:
                alternative = 'less' if col in min_cols else 'greater'
                p = stat_test(df.loc[[best], col], df.loc[[i], col], alternative=alternative).pvalue
                if p >= threshold:
                    mean.loc[i, col] = '\\textbf{' + mean.loc[i, col] + '}'

    return mean
