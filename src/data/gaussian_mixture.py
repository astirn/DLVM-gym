import matplotlib.pyplot as plt
import seaborn as sns
import torch
from toolz import merge_with
from torch.distributions import Categorical, Dirichlet, MultivariateNormal, Wishart
from torch.utils.data import TensorDataset

from .abstract import AbstractDataModule


def collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    rows = [{"x": b[0], "c": b[1]} for b in batch]
    out: dict[str, torch.Tensor] = merge_with(torch.vstack, *rows)
    out["c"] = out["c"].reshape(-1).long()
    return out


class GaussianMixtureDataModule(AbstractDataModule):

    def __init__(
        self,
        num_clusters: int,
        num_samples: dict[str, int],
        seed: int,
        **kwargs,
    ) -> None:
        super().__init__(collate_fn=collate, **kwargs)
        self.num_clusters = num_clusters
        self.num_samples = num_samples
        self.seed = seed
        self._proportions: torch.Tensor | None = None
        self._means: torch.Tensor | None = None
        self._covariances: torch.Tensor | None = None

    def prepare_data(self) -> None: ...

    @property
    def proportions(self) -> torch.Tensor:
        if self._proportions is None:
            torch.manual_seed(self.seed)
            self._proportions = Dirichlet(3 * torch.ones(self.num_clusters)).sample()
        return self._proportions

    @property
    def means(self) -> torch.Tensor:
        if self._means is None:
            torch.manual_seed(self.seed)
            self._means = torch.randn(self.num_clusters, 2) * 3
        return self._means

    @property
    def covariances(self) -> torch.Tensor:
        if self._covariances is None:
            torch.manual_seed(self.seed)
            self._covariances = Wishart(3, scale_tril=torch.eye(2) / 3).sample(
                torch.Size([self.num_clusters])
            )
        return self._covariances

    def _sample_mixture(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        if n == 0:
            return torch.empty(0, 2), torch.empty(0)

        c = Categorical(probs=self.proportions).sample(torch.Size([n]))
        x_c = MultivariateNormal(
            loc=self.means, covariance_matrix=self.covariances
        ).sample(torch.Size([n]))
        x = x_c[torch.nn.functional.one_hot(c, num_classes=self.num_clusters).bool()]
        return x, c.long()

    def setup(self, stage: str | None = None) -> None:
        self._batch_size_per_device()

        fit = stage == "fit" or stage is None
        validate = stage == "validate" or stage is None
        test_or_predict = stage == "test" or stage == "predict" or stage is None

        if fit and self.data["train"] is None:
            torch.manual_seed(self.seed + 1)
            self.data["train"] = TensorDataset(
                *self._sample_mixture(self.num_samples["train"])
            )

        if (fit or validate) and self.data["val"] is None:
            torch.manual_seed(self.seed + 2)
            self.data["val"] = TensorDataset(
                *self._sample_mixture(self.num_samples["val"])
            )

        if test_or_predict and self.data["test"] is None:
            torch.manual_seed(self.seed + 3)
            self.data["test"] = TensorDataset(
                *self._sample_mixture(self.num_samples["test"])
            )


if __name__ == "__main__":
    dm = GaussianMixtureDataModule(
        num_clusters=3, num_samples={"train": 1_000, "val": 1_000, "test": 0}, seed=0
    )
    dm.setup()

    x, c = zip(*[(b["x"], b["c"]) for b in dm.train_dataloader()])
    x = torch.cat(x, dim=0).numpy()
    c = torch.cat(c, dim=0).numpy()

    fig, ax = plt.subplots()
    fig.suptitle("Gaussian Mixture")
    sns.scatterplot(
        x=x[:, 0],
        y=x[:, 1],
        hue=c,
        palette="deep",
        legend=False,
        ax=ax,
        alpha=0.5,
        s=10,
    )
    ax.set_aspect("equal")
    plt.show()
