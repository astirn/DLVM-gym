import logging
from collections.abc import Callable

import pandas as pd
import torch
from anndata import AnnData
from toolz import merge_with
from torchvision import datasets
from torchvision.transforms import ToTensor

from .abstract import AbstractDataModule

logger = logging.getLogger(__name__)


def collate(batch: list) -> dict[str, torch.Tensor]:
    rows = [{"x": b[0], "c": torch.tensor(b[1]).unsqueeze(0)} for b in batch]
    out = merge_with(torch.vstack, *rows)
    out["x"] = (out["x"].flatten(start_dim=1) - 0.5) * 2
    out["c"] = out["c"].reshape(-1).long()
    return out


def rename_columns(adata: AnnData, columns: dict[str, str]) -> AnnData:
    if isinstance(adata.obs, pd.DataFrame):
        adata.obs = adata.obs.rename(columns=columns)
    else:
        raise ValueError("adata.obs is not a pandas DataFrame")
    return adata


class scRNADataModule(AbstractDataModule):

    def __init__(
        self, dataset: str, prepare_data_fn: Callable[[str], AnnData], **kwargs
    ) -> None:
        super().__init__(collate_fn=collate, **kwargs)
        self.dataset = dataset
        self.prepare_data_fn = prepare_data_fn
        self.root = f"datasets/scRNA-seq/{dataset}"

        self._adata: AnnData | None = None
        self._dim: int | None = None
        self._num_data: int | None = None

    @property
    def dim(self) -> int | None:
        if self._dim is None:
            logger.warning(
                "Dimension is only set after prepare_data is called, returning None"
            )
            return None
        return self._dim

    @property
    def num_data(self) -> int | None:
        if self._num_data is None:
            logger.warning(
                "Number of data is only set after prepare_data is called, returning None"
            )
            return None
        return self._num_data

    def prepare_data(self) -> None:
        self._adata = self.prepare_data_fn(self.root)

    def setup(self, stage: str | None = None) -> None:
        self._batch_size_per_device()

        fit = stage == "fit" or stage is None
        validate = stage == "validate" or stage is None
        test_or_predict = stage in ("test", "predict") or stage is None

        if fit and self.data["train"] is None:
            self.data["train"] = datasets.MNIST(
                root=self.root, train=True, transform=ToTensor()
            )

        if (fit or validate) and self.data["val"] is None:
            self.data["val"] = datasets.MNIST(
                root=self.root, train=False, transform=ToTensor()
            )

        if test_or_predict and self.data["test"] is None:
            self.data["test"] = datasets.MNIST(
                root=self.root, train=False, transform=ToTensor()
            )
