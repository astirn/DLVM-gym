from typing import cast

import pandas as pd
import torch
from anndata import AnnData
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
from torchmetrics import Metric

from src.models.abstract import DeepLatentVariableModel


class scIBMetrics(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("_batch", default=[])
        self.add_state("_cell_type", default=[])
        self.add_state("_embedding", default=[])
        self.add_state("_x_pca", default=[])

    @property
    def batch(self) -> list[torch.Tensor]:
        return cast(list[torch.Tensor], self._batch)

    @property
    def cell_type(self) -> list[torch.Tensor]:
        return cast(list[torch.Tensor], self._cell_type)

    @property
    def embedding(self) -> list[torch.Tensor]:
        return cast(list[torch.Tensor], self._embedding)

    @property
    def x_pca(self) -> list[torch.Tensor]:
        return cast(list[torch.Tensor], self._x_pca)

    def update(
        self, batch: dict[str, torch.Tensor], model: DeepLatentVariableModel
    ) -> None:
        c, s = batch["c"], batch["s"]
        z = model.qz(**batch).mean
        self.batch.append(s)
        self.cell_type.append(c)
        self.embedding.append(z)
        if "x_pca" in batch:
            self.x_pca.append(batch["x_pca"])

    def _build_adata(self) -> tuple[AnnData, str | None]:
        embedding = torch.cat(self.embedding, dim=0)
        obs = pd.DataFrame(
            {
                "batch": torch.cat(self.batch).numpy(force=True),
                "cell_type": torch.cat(self.cell_type).numpy(force=True),
            }
        )
        adata = AnnData(X=embedding.numpy(force=True), obs=obs)
        adata.obsm["embedding"] = embedding.numpy(force=True)

        pre_integrated_key = None
        if len(self.x_pca) > 0:
            pre_integrated_key = "X_pca"
            adata.obsm[pre_integrated_key] = torch.cat(self.x_pca, dim=0).numpy(
                force=True
            )
        return adata, pre_integrated_key

    def _bio_conservation_metrics(self, data: AnnData) -> BioConservation:
        return BioConservation(
            isolated_labels=True,
            nmi_ari_cluster_labels_leiden=True,
            nmi_ari_cluster_labels_kmeans=True,
            silhouette_label=True,
            clisi_knn=True,
        )

    def _batch_correction_metrics(self, data: AnnData) -> BatchCorrection:
        if data.obs["batch"].nunique() > 1:
            return BatchCorrection(
                bras=True,
                ilisi_knn=True,
                kbet_per_label=True,
                graph_connectivity=True,
                pcr_comparison=True,
            )
        else:
            return BatchCorrection(
                bras=False,
                ilisi_knn=False,
                kbet_per_label=False,
                graph_connectivity=False,
                pcr_comparison=True,  # this is a hack to make the metrics work with just a single batch
            )

    def _benchmark(self, adata: AnnData, pre_integrated_key: str | None) -> pd.Series:
        bm = Benchmarker(
            adata,
            batch_key="batch",
            label_key="cell_type",
            embedding_obsm_keys=["embedding"],
            bio_conservation_metrics=self._bio_conservation_metrics(adata),
            batch_correction_metrics=self._batch_correction_metrics(adata),
            pre_integrated_embedding_obsm_key=pre_integrated_key,
            n_jobs=-1,
        )
        bm.benchmark()

        df = bm.get_results(min_max_scale=False)
        metric_type_row = df.loc["Metric Type"]
        df = df.drop(index="Metric Type")
        df.columns = pd.MultiIndex.from_arrays(
            [metric_type_row.values, df.columns], names=["Metric Group", "Metric Name"]
        )
        df = df.astype(float)
        return df.iloc[0]

    def _to_metric_dict(self, row: pd.Series) -> dict[str, torch.Tensor]:
        return {
            f"{group}/{name}": torch.tensor(val, dtype=torch.float32)
            for (group, name), val in dict(row).items()
        }

    def compute(self) -> dict[str, torch.Tensor]:
        adata, pre_integrated_key = self._build_adata()
        row = self._benchmark(adata, pre_integrated_key)
        return self._to_metric_dict(row)
