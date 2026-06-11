from math import prod

import torch
from toolz import merge_with
from torchvision import datasets
from torchvision.transforms import ToTensor

from .base import BaseDataModule


def collate(batch: list) -> dict[str, torch.Tensor]:
    rows = [{"x": b[0], "c": torch.tensor(b[1]).unsqueeze(0)} for b in batch]
    out = merge_with(torch.vstack, *rows)
    out["x"] = (out["x"].flatten(start_dim=1) - 0.5) * 2
    out["c"] = out["c"].reshape(-1).long()
    return out


class ClusteringDataModule(BaseDataModule):

    def __init__(self, dataset: str, **kwargs) -> None:
        super().__init__(collate_fn=collate, **kwargs)
        self.dataset = dataset
        self.root = f"datasets/clustering"

        self._dim: int | None = None
        self._num_data: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise ValueError("Property is not set. Call prepare_data first.")
        return self._dim

    @property
    def num_data(self) -> int:
        if self._num_data is None:
            raise ValueError("Property is not set. Call prepare_data first.")
        return self._num_data

    def prepare_data(self) -> None:
        train = getattr(datasets, self.dataset)(
            root=self.root, train=True, download=True
        )
        getattr(datasets, self.dataset)(root=self.root, train=False, download=True)
        self._dim = prod(train.data[0].shape)
        self._num_data = len(train)

    def setup(self, stage: str | None = None) -> None:
        self._batch_size_per_device()

        fit = stage == "fit" or stage is None
        validate = stage == "validate" or stage is None
        test_or_predict = stage in ("test", "predict") or stage is None

        if fit and self.data["train"] is None:
            self.data["train"] = getattr(datasets, self.dataset)(
                root=self.root, train=True, transform=ToTensor()
            )

        if (fit or validate) and self.data["val"] is None:
            self.data["val"] = getattr(datasets, self.dataset)(
                root=self.root, train=False, transform=ToTensor()
            )

        if test_or_predict and self.data["test"] is None:
            self.data["test"] = getattr(datasets, self.dataset)(
                root=self.root, train=False, transform=ToTensor()
            )
