import pandas as pd
import torch
from anndata import AnnData
from scib_metrics.benchmark import BatchCorrection, Benchmarker
from torchmetrics import Metric

from src.models.abstract import DeepLatentVariableModel


def batch_correction_metrics(data: AnnData) -> BatchCorrection:
    if data.obs["batch"].nunique() > 1:
        return BatchCorrection(
            bras=True,
            ilisi_knn=True,
            kbet_per_label=True,
            graph_connectivity=True,
            pcr_comparison=False,
        )
    else:
        return BatchCorrection(
            bras=False,
            ilisi_knn=False,
            kbet_per_label=False,
            graph_connectivity=False,
            pcr_comparison=True,  # this is a hack to make the metrics work with just a single batch
        )


class scIBMetrics(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("batch", default=[])
        self.add_state("cell_type", default=[])
        self.add_state("embedding", default=[])

    def update(
        self, batch: dict[str, torch.Tensor], model: DeepLatentVariableModel
    ) -> None:
        c, s = batch["c"], batch["s"]
        z = model.qz(**batch).sample()
        self.batch.append(s)  # type: ignore[attr-defined]
        self.cell_type.append(c)  # type: ignore[attr-defined]
        self.embedding.append(z)  # type: ignore[attr-defined]

    def compute(self) -> dict[str, torch.Tensor]:
        embedding = torch.cat(self.embedding, dim=0)  # type: ignore[attr-defined]
        obs = pd.DataFrame(
            {
                "batch": torch.cat(self.batch).numpy(force=True),  # type: ignore[attr-defined]
                "cell_type": torch.cat(self.cell_type).numpy(force=True),  # type: ignore[attr-defined]
            }
        )
        adata = AnnData(X=embedding.numpy(force=True), obs=obs)
        adata.obsm["embedding"] = embedding.numpy(force=True)

        bm = Benchmarker(
            adata,
            batch_key="batch",
            label_key="cell_type",
            embedding_obsm_keys=["embedding"],
            batch_correction_metrics=batch_correction_metrics(adata),
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
        row = df.iloc[0]

        return {
            f"{group}/{name}": torch.tensor(val, dtype=torch.float32)
            for (group, name), val in row.items()
        }
