import torch
from toolz import merge_with
from torchvision import datasets
from torchvision.transforms import ToTensor

from .abstract import AbstractDataModule


def collate(batch: list) -> dict[str, torch.Tensor]:
    rows = [{"x": b[0], "c": torch.tensor(b[1]).unsqueeze(0)} for b in batch]
    out = merge_with(torch.vstack, *rows)
    out["x"] = (out["x"].flatten(start_dim=1) - 0.5) * 2
    out["c"] = out["c"].reshape(-1).long()
    return out


class MNISTDataModule(AbstractDataModule):

    def __init__(self, **kwargs) -> None:
        super().__init__(collate_fn=collate, **kwargs)
        self.root = "datasets"

    @property
    def dim(self) -> int:
        return 784

    @property
    def num_data(self) -> int:
        return 60_000

    def prepare_data(self) -> None:
        datasets.MNIST(root=self.root, train=True, download=True)
        datasets.MNIST(root=self.root, train=False, download=True)

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
