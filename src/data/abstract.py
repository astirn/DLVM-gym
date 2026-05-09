import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def worker_init_fn(worker_id: int):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ.update(
        {
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PL_MAX_THREADS": "1",
        }
    )


class AbstractDataModule(LightningDataModule, ABC):
    """Abstract data module for loading and splitting data."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        batch_size: int | dict[str, int] = kwargs.get("batch_size", 32)
        if isinstance(batch_size, int):
            batch_size = {"train": batch_size, "val": batch_size, "test": batch_size}
        self.batch_size: dict[str, int] = batch_size
        self.num_workers: int | dict[str, int] = kwargs.get("num_workers", 0)
        self.pin_memory: bool = kwargs.get("pin_memory", False)
        self.collate_fn: Callable | None = kwargs.get("collate_fn", None)

        self.data: dict[str, Dataset | None] = {
            "train": None,
            "val": None,
            "test": None,
        }

        self.batch_size_per_device = {s: self.batch_size[s] for s in self.data}

    @abstractmethod
    def prepare_data(self) -> None: ...

    @abstractmethod
    def setup(self, stage: str | None = None) -> None: ...

    def _batch_size_per_device(self) -> None:
        if self.trainer is not None:
            for split, batch_size in self.batch_size.items():
                if batch_size % self.trainer.world_size != 0:
                    raise RuntimeError(
                        f"Batch size ({batch_size}) for split {split} is not divisible by number of devices ({self.trainer.world_size})."
                    )
                self.batch_size_per_device[split] = (
                    batch_size // self.trainer.world_size
                )

    def _dl_kwargs(self, split: str) -> dict[str, Any]:
        num_workers = (
            self.num_workers
            if isinstance(self.num_workers, int)
            else self.num_workers[split]
        )
        out = {
            "batch_size": self.batch_size_per_device[split],
            "num_workers": num_workers,
            "pin_memory": self.pin_memory,
            "worker_init_fn": worker_init_fn if num_workers > 0 else None,
        }
        if self.collate_fn is not None:
            out["collate_fn"] = self.collate_fn
        return out

    def train_dataloader(self) -> DataLoader:
        if self.data["train"] is None:
            raise ValueError("No training dataset found")
        return DataLoader(
            dataset=self.data["train"],
            shuffle=True,
            drop_last=True,
            **self._dl_kwargs(split="train"),
        )

    def val_dataloader(self) -> DataLoader:
        if self.data["val"] is None:
            raise ValueError("No validation dataset found")
        return DataLoader(
            dataset=self.data["val"],
            shuffle=False,
            drop_last=False,
            **self._dl_kwargs(split="val"),
        )

    def test_dataloader(self) -> DataLoader:
        if self.data["test"] is None:
            raise ValueError("No test dataset found")
        return DataLoader(
            dataset=self.data["test"],
            shuffle=False,
            drop_last=False,
            **self._dl_kwargs(split="test"),
        )

    def predict_dataloader(self) -> DataLoader:
        if self.data["test"] is None:
            raise ValueError("No test dataset found")
        return DataLoader(
            dataset=self.data["test"],
            shuffle=False,
            drop_last=False,
            **self._dl_kwargs(split="test"),
        )
