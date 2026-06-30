from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import torch
from anndata import AnnData
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .base import BaseDataModule


class scRNADataset(Dataset):

    def __init__(
        self,
        counts: sp.csr_matrix,
        batch_codes: torch.Tensor,
        cell_type_codes: torch.Tensor,
    ) -> None:
        self.counts = counts
        self.batch_codes = batch_codes
        self.cell_type_codes = cell_type_codes

    def __len__(self) -> int:
        return self.counts.shape[0]  # type: ignore[attr-defined]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.counts[idx].todense().astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "c": self.cell_type_codes[idx].long(),
            "s": self.batch_codes[idx].long(),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.vstack([b["x"] for b in batch]),
        "l": torch.hstack([b["x"].sum(dim=1) for b in batch]),
        "c": torch.hstack([b["c"] for b in batch]),
        "s": torch.hstack([b["s"] for b in batch]),
    }


class scRNADataModule(BaseDataModule):

    def __init__(
        self,
        dataset: str,
        prepare_data_fn: Callable[[str], AnnData],
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(collate_fn=collate, **kwargs)
        self.dataset = dataset
        self.root = f"datasets/scRNA-seq/{dataset}"
        self.prepare_data_fn = prepare_data_fn

        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self._counts: sp.csr_matrix | None = None
        self._batch_keys: list[str] | None = None
        self._batch_codes: torch.Tensor | None = None
        self._cell_type_keys: list[str] | None = None
        self._cell_type_codes: torch.Tensor | None = None

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
    def num_cells(self) -> int:
        return self.counts.shape[0]  # type: ignore[attr-defined]

    @property
    def num_genes(self) -> int:
        return self.counts.shape[1]  # type: ignore[attr-defined]

    def prepare_data(self) -> None:
        adata = self.prepare_data_fn(self.root)
        self._counts = sp.csr_matrix(adata.X, dtype=np.int16)
        for covariate in ["batch", "cell_type"]:
            keys, codes = np.unique(adata.obs[covariate], return_inverse=True)
            setattr(self, f"_{covariate}_keys", list(map(str, keys)))
            setattr(self, f"_{covariate}_codes", torch.tensor(codes, dtype=torch.int8))

    def setup(self, stage: str | None = None) -> None:
        self._batch_size_per_device()

        fit = stage == "fit" or stage is None
        validate = stage == "validate" or stage is None
        test = stage == "test" or stage is None
        predict = stage == "predict" or stage is None

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(self.num_cells)

        n_val = int(self.num_cells * self.val_ratio)
        n_test = int(self.num_cells * self.test_ratio)
        n_train = self.num_cells - n_val - n_test

        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]
        predict_idx = np.arange(self.num_cells)

        if fit and self.data["train"] is None:
            self.data["train"] = scRNADataset(
                self.counts[train_idx],
                self.batch_codes[train_idx],
                self.cell_type_codes[train_idx],
            )

        if (fit or validate) and self.data["val"] is None:
            self.data["val"] = scRNADataset(
                self.counts[val_idx],
                self.batch_codes[val_idx],
                self.cell_type_codes[val_idx],
            )

        if test and self.data["test"] is None:
            self.data["test"] = scRNADataset(
                self.counts[test_idx],
                self.batch_codes[test_idx],
                self.cell_type_codes[test_idx],
            )

        if predict and self.data["predict"] is None:
            self.data["predict"] = scRNADataset(
                self.counts[predict_idx],
                self.batch_codes[predict_idx],
                self.cell_type_codes[predict_idx],
            )

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
