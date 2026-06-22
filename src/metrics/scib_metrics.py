import pandas as pd
from anndata import AnnData
from scib_metrics.benchmark import BatchCorrection, Benchmarker


def scib_metrics(data: AnnData, embedding_obsm_keys: list[str]) -> pd.DataFrame:

    if data.obs["batch"].nunique() > 1:
        batch_correction_metrics = BatchCorrection(
            bras=True,
            ilisi_knn=True,
            kbet_per_label=True,
            graph_connectivity=True,
            pcr_comparison=False,
        )
    else:
        batch_correction_metrics = BatchCorrection(
            bras=False,
            ilisi_knn=False,
            kbet_per_label=False,
            graph_connectivity=False,
            pcr_comparison=True,  # this is a hack to make the metrics work with just a single batch
        )

    bm = Benchmarker(
        data,
        batch_key="batch",
        label_key="cell_type",
        embedding_obsm_keys=embedding_obsm_keys,
        batch_correction_metrics=batch_correction_metrics,
        n_jobs=-1,
    )
    bm.benchmark()

    df = bm.get_results()
    metric_type_row = df.loc["Metric Type"]
    df = df.drop(index="Metric Type")
    df.columns = pd.MultiIndex.from_arrays(
        [metric_type_row.values, df.columns], names=["Metric Group", "Metric Name"]
    )
    df = df.astype(float)

    return df
