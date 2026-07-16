from typing import cast

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from scib_metrics import silhouette_batch
from scib_metrics.benchmark import BatchCorrection
from scipy.sparse.csgraph import connected_components
from toolz import keyfilter, keymap

from .scib_metrics import scIBMetrics

OP_METRIC_MAP: dict[str, str] = {
    "Leiden NMI": "nmi",
    "Leiden ARI": "ari",
    "Silhouette label": "asw_label",
    "cLISI": "clisi",
    "Isolated labels": "isolated_label_asw",
    "iLISI": "ilisi",
    "KBET": "kbet",
    "PCR comparison": "pcr",
}


def _op_graph_connectivity(
    embedding: np.ndarray, labels: np.ndarray, *, n_neighbors: int = 15
) -> float:
    """OpenProblems ``scib.metrics.graph_connectivity``.

    Builds a Scanpy neighborhood graph on the embedding and scores the mean
    fraction of cells in the largest connected component of each cell-type
    subgraph of ``obsp['connectivities']``.
    """
    adata = AnnData(X=embedding.copy())
    adata.obsm["X_emb"] = embedding
    adata.obs["cell_type"] = pd.Categorical(labels)
    sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=n_neighbors)

    clust_res: list[float] = []
    for label in adata.obs["cell_type"].cat.categories:
        adata_sub = adata[adata.obs["cell_type"].isin([label])]
        _, comps = connected_components(
            adata_sub.obsp["connectivities"], connection="strong"
        )
        tab = pd.Series(comps).value_counts()
        clust_res.append(tab.max() / sum(tab))
    return float(np.mean(clust_res))


class OpenProblemsMetrics(scIBMetrics):
    """scIB metrics with OpenProblems naming, classic ``asw_batch``, and OP GC."""

    def _batch_correction_metrics(self, data: AnnData) -> BatchCorrection:
        if data.obs["batch"].nunique() > 1:
            return BatchCorrection(
                bras=False,  # OpenProblems uses ASW batch instead of BRAS
                ilisi_knn=True,
                kbet_per_label=True,
                graph_connectivity=False,  # OpenProblems method differs from scIB
                pcr_comparison=True,
            )
        return BatchCorrection(
            bras=False,
            ilisi_knn=False,
            kbet_per_label=False,
            graph_connectivity=False,
            pcr_comparison=True,  # this is a hack to make the metrics work with just a single batch
        )

    def _to_metric_dict(self, row: pd.Series) -> dict[str, torch.Tensor]:
        metrics = super()._to_metric_dict(row)
        metrics = keymap(lambda k: k.split("/")[1], metrics)
        metrics = keyfilter(lambda k: k in OP_METRIC_MAP.keys(), metrics)
        metrics = keymap(lambda k: OP_METRIC_MAP[k], metrics)
        return metrics

    def compute(self) -> dict[str, torch.Tensor]:
        adata, pre_integrated_key = self._build_adata()
        row = self._benchmark(adata, pre_integrated_key)
        metrics = self._to_metric_dict(row)

        embedding = cast(np.ndarray, adata.obsm["embedding"])
        labels = adata.obs["cell_type"].to_numpy()

        metrics["graph_connectivity"] = torch.tensor(
            _op_graph_connectivity(embedding, labels), dtype=torch.float32
        )

        if adata.obs["batch"].nunique() > 1:
            asw = silhouette_batch(
                embedding,
                labels,
                adata.obs["batch"].to_numpy(),
                metric="euclidean",
                between_cluster_distances="nearest",
            )
            metrics["asw_batch"] = torch.tensor(asw, dtype=torch.float32)

        metrics = dict(sorted(metrics.items()))

        return keymap(lambda x: f"op/{x}", metrics)
