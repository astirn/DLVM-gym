from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .base import BaseDataModule
from .utils import compute_preintegration_pca


class scRNADataset(Dataset):

    def __init__(
        self,
        counts: sp.csr_matrix,
        batch_codes: torch.Tensor,
        cell_type_codes: torch.Tensor,
        group_codes: torch.Tensor,
        pca: np.ndarray | None = None,
    ) -> None:
        self.counts = counts
        self.batch_codes = batch_codes
        self.cell_type_codes = cell_type_codes
        self.group_codes = group_codes
        self.pca = pca

    def __len__(self) -> int:
        return self.counts.shape[0]  # type: ignore[attr-defined]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.counts[idx].todense().astype(np.float32)
        item = {
            "x": torch.from_numpy(x),
            "c": self.cell_type_codes[idx].long(),
            "s": self.batch_codes[idx].long(),
            "g": self.group_codes[idx].long(),
        }
        if self.pca is not None:
            item["x_pca"] = torch.from_numpy(self.pca[idx].astype(np.float32))
        return item


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {
        "x": torch.vstack([b["x"] for b in batch]),
        "l": torch.hstack([b["x"].sum(dim=1) for b in batch]),
        "c": torch.hstack([b["c"] for b in batch]),
        "s": torch.hstack([b["s"] for b in batch]),
        "g": torch.hstack([b["g"] for b in batch]),
    }
    if "x_pca" in batch[0]:
        out["x_pca"] = torch.vstack([b["x_pca"] for b in batch])
    return out


class scRNADataModule(BaseDataModule):

    def __init__(
        self,
        dataset: str,
        prepare_data_fn: Callable[[str], AnnData],
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int | None = None,
        group_covariate: str = "batch",
        **kwargs,
    ) -> None:
        super().__init__(collate_fn=collate, **kwargs)
        self.dataset = dataset
        self.root = f"datasets/scRNA-seq/{dataset}"
        self.prepare_data_fn = prepare_data_fn

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.group_covariate = group_covariate

        self._counts: sp.csr_matrix | None = None
        self._batch_keys: list[str] | None = None
        self._batch_codes: torch.Tensor | None = None
        self._cell_type_keys: list[str] | None = None
        self._cell_type_codes: torch.Tensor | None = None
        self._group_keys: list[str] | None = None
        self._group_codes: torch.Tensor | None = None
        self._pca: np.ndarray | None = None

    @property
    def adata(self) -> AnnData:
        return self.prepare_data_fn(self.root)

    @property
    def counts(self) -> sp.csr_matrix:
        if self._counts is None:
            raise ValueError("Property is not set. Call prepare_data first.")
        return self._counts

    @property
    def batch_codes(self) -> torch.Tensor:
        if self._batch_codes is None:
            raise ValueError("Property is not set. Call prepare_data first.")
        return self._batch_codes

    @property
    def batch_keys(self) -> list[str]:
        if self._batch_keys is None:
            raise ValueError("Property is not set. Call prepare_data first.")
        return self._batch_keys

    @property
    def cell_type_codes(self) -> torch.Tensor:
        if self._cell_type_codes is None:
            raise ValueError("Property is not set. Call prepare_data first.")
        return self._cell_type_codes

    @property
    def cell_type_keys(self) -> list[str]:
        if self._cell_type_keys is None:
            raise ValueError("Property is not set. Call prepare_data first.")
        return self._cell_type_keys

    @property
    def group_codes(self) -> torch.Tensor:
        if self._group_codes is None:
            raise ValueError("Property is not set. Call prepare_data first.")
        return self._group_codes

    @property
    def group_keys(self) -> list[str]:
        if self._group_keys is None:
            raise ValueError("Property is not set. Call prepare_data first.")
        return self._group_keys

    @property
    def num_groups(self) -> int:
        return len(self.group_keys)

    @property
    def pca(self) -> np.ndarray:
        if self._pca is None:
            raise ValueError("Property is not set. Call prepare_data first.")
        return self._pca

    @property
    def num_data(self) -> int:
        return self.counts.shape[0]  # type: ignore[attr-defined]

    @property
    def num_genes(self) -> int:
        return self.counts.shape[1]  # type: ignore[attr-defined]

    def prepare_data(self) -> None:
        adata = self.prepare_data_fn(self.root)
        counts = sp.csr_matrix(adata.X)
        if "X_pca" in adata.obsm:
            self._pca = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
        else:
            self._pca = compute_preintegration_pca(counts)
        if np.max(counts.data) < 2**15 - 1:
            self._counts = counts.astype(np.int16)
        else:
            self._counts = counts.astype(np.int32)
        for attr, covariate in [
            ("batch", "batch"),
            ("cell_type", "cell_type"),
            ("group", self.group_covariate),
        ]:
            keys, codes = np.unique(adata.obs[covariate], return_inverse=True)
            setattr(self, f"_{attr}_keys", list(map(str, keys)))
            setattr(self, f"_{attr}_codes", torch.tensor(codes, dtype=torch.int32))

    def _dataset(self, idx: np.ndarray) -> scRNADataset:
        return scRNADataset(
            self.counts[idx],
            self.batch_codes[idx],
            self.cell_type_codes[idx],
            self.group_codes[idx],
            self.pca[idx],
        )

    def setup(self, stage: str | None = None) -> None:
        self._batch_size_per_device()

        fit = stage == "fit" or stage is None
        validate = stage == "validate" or stage is None
        test = stage == "test" or stage is None
        predict = stage == "predict" or stage is None

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(self.num_data)

        n_val = int(self.num_data * self.val_ratio)
        n_test = int(self.num_data * self.test_ratio)
        n_train = self.num_data - n_val - n_test

        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        predict_idx = np.arange(self.num_data)

        if fit and self.data["train"] is None:
            self.data["train"] = self._dataset(train_idx)

        if (fit or validate) and self.data["val"] is None:
            self.data["val"] = self._dataset(val_idx)

        if test and self.data["test"] is None:
            self.data["test"] = self._dataset(test_idx)

        if predict and self.data["predict"] is None:
            self.data["predict"] = self._dataset(predict_idx)

    def full_dataloader(self) -> DataLoader:
        return DataLoader(
            ConcatDataset(
                [
                    self.train_dataloader().dataset,
                    self.val_dataloader().dataset,
                    self.test_dataloader().dataset,
                ]
            ),
            shuffle=False,
            **self._dl_kwargs(split="val"),
        )
