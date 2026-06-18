import torch
from toolz import merge_with
from torch.distributions import Categorical, Dirichlet, MultivariateNormal, Wishart
from torch.utils.data import TensorDataset

from .base import BaseDataModule


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    rows = [{"x": b[0], "c": b[1]} for b in batch]
    out: dict[str, torch.Tensor] = merge_with(torch.vstack, *rows)
    out["c"] = out["c"].reshape(-1).long()
    return out


class GaussianMixtureDataModule(BaseDataModule):

    def __init__(self, num_clusters: int, num_data: int, seed: int, **kwargs) -> None:
        super().__init__(collate_fn=collate, **kwargs)
        self._num_clusters = num_clusters
        self._num_data = num_data
        self._seed = seed
        self._proportions: torch.Tensor | None = None
        self._means: torch.Tensor | None = None
        self._covariances: torch.Tensor | None = None

    @property
    def dim(self) -> int:
        return 2

    @property
    def num_data(self) -> int:
        return self._num_data

    def prepare_data(self) -> None:
        torch.manual_seed(self._seed)
        if self._proportions is None:
            self._proportions = Dirichlet(3 * torch.ones(self._num_clusters)).sample()
        if self._means is None:
            self._means = torch.randn(self._num_clusters, 2) * 3
        if self._covariances is None:
            self._covariances = Wishart(3, scale_tril=torch.eye(2) / 3).sample(
                torch.Size([self._num_clusters])
            )

    def _sample_mixture(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._proportions is None
            or self._means is None
            or self._covariances is None
        ):
            raise ValueError("Properties are not set. Call prepare_data first.")

        if n == 0:
            return torch.empty(0, 2), torch.empty(0)

        c = Categorical(probs=self._proportions).sample(torch.Size([n]))
        x_c = MultivariateNormal(
            loc=self._means, covariance_matrix=self._covariances
        ).sample(torch.Size([n]))
        x = x_c[torch.nn.functional.one_hot(c, num_classes=self._num_clusters).bool()]

        return x, c.long()

    def setup(self, stage: str | None = None) -> None:
        self._batch_size_per_device()

        if (stage == "fit" or stage is None) and self.data["train"] is None:
            torch.manual_seed(self._seed + 1)
            self.data["train"] = TensorDataset(*self._sample_mixture(self.num_data))

        if (stage in ("fit", "validate") or stage is None) and self.data["val"] is None:
            torch.manual_seed(self._seed + 2)
            self.data["val"] = TensorDataset(*self._sample_mixture(self.num_data))

        if (stage == "test" or stage is None) and self.data["test"] is None:
            torch.manual_seed(self._seed + 3)
            self.data["test"] = TensorDataset(*self._sample_mixture(self.num_data))

        if (stage == "predict" or stage is None) and self.data["predict"] is None:
            torch.manual_seed(self._seed + 4)
            self.data["predict"] = TensorDataset(*self._sample_mixture(self.num_data))
