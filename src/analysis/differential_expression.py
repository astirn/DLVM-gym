from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning import Trainer
from toolz import merge_with

from src.data.scRNA import scRNADataModule
from src.distributions import concat_distributions
from src.model_wrapper import ModelWrapper


def differential_expression(
    datamodule: scRNADataModule,
    model: ModelWrapper,
    trainer: Trainer,
    save_path: str,
    *,
    condition_column: str,
    group1: str,
    group2: str,
    fig_name: str,
):
    # get model predictions
    predictions = trainer.predict(model, datamodule=datamodule)
    predictions = merge_with(concat_distributions, predictions)

    # normalize predictions and observed counts
    y_hat = predictions["px"].mu
    y = torch.sparse_csr_tensor(
        torch.tensor(datamodule.adata.X.indptr, dtype=torch.int64),
        torch.tensor(datamodule.adata.X.indices, dtype=torch.int64),
        torch.tensor(datamodule.adata.X.data, dtype=torch.float32),
        size=datamodule.adata.X.shape,
        device=y_hat.device,
    ).to_dense()
    l = y.sum(axis=1, keepdims=True)
    y = y / l
    y_hat = y_hat / l

    # compute mean expression for index
    df_index = datamodule.adata.obs[[condition_column] + ["batch", "cell_type"]]
    df_index = df_index.set_index([condition_column] + ["batch", "cell_type"])
    df_mean_obs = pd.DataFrame()
    df_mean_pred = pd.DataFrame()
    for index in df_index.index.unique():
        sample_index = df_index.index == index

        mean_obs = pd.DataFrame(
            y[sample_index, :].mean(axis=0, keepdims=True).numpy(force=True),
            index=pd.MultiIndex.from_tuples([index]),
        )
        mean_pred = pd.DataFrame(
            y_hat[sample_index, :].mean(axis=0, keepdims=True).numpy(force=True),
            index=pd.MultiIndex.from_tuples([index]),
        )
        df_mean_obs = pd.concat([df_mean_obs, mean_obs])
        df_mean_pred = pd.concat([df_mean_pred, mean_pred])

    # compute conditional differential expression between the two groups
    de_obs = df_mean_obs.loc[group1] - df_mean_obs.loc[group2]
    de_pred = df_mean_pred.loc[group1] - df_mean_pred.loc[group2]
    de_obs.dropna(axis=0, how="any", inplace=True)
    de_pred.dropna(axis=0, how="any", inplace=True)

    de_obs = de_obs.values.reshape(-1)
    np.save(Path(save_path) / f"de_obs.npy", de_obs)
    de_pred = de_pred.values.reshape(-1)
    np.save(Path(save_path) / f"de_pred.npy", de_pred)

    # plot all conditional differential expression
    plt.figure(figsize=(8, 8))
    sns.regplot(x=de_obs, y=de_pred, scatter_kws={"rasterized": True})
    plt.xlabel("Observed DE")
    plt.ylabel("Predicted DE")
    plt.title(f"Differential Expression: {group1} vs {group2}")
    plt.text(
        0.05,
        0.95,
        f"Pearson r = {np.corrcoef(de_obs, de_pred)[0, 1]:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )
    plt.savefig(Path(save_path) / fig_name)
    plt.close()
