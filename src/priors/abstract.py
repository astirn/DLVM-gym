from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor
from torch.distributions import Categorical, Distribution
from torch.nn import Module


class Prior(Module, ABC):

    @property
    def has_global_latent_variables(self) -> bool:
        return False

    @property
    def alternating_inference(self) -> bool:
        return False

    def empirical_bayes_params(self) -> list[Tensor]:
        raise NotImplementedError  # required when alternating_inference is True

    def empirical_bayes_loss(self, batch: dict[str, Any]) -> Tensor:
        raise NotImplementedError  # required when alternating_inference is True

    @property
    @abstractmethod
    def pz(self) -> Distribution: ...

    @abstractmethod  # required for model training
    def dkl(self, qz_x: Distribution, z_sample: Tensor) -> Tensor: ...

    def cluster_responsibilities(self, z: Tensor) -> Categorical:
        raise NotImplementedError  # optionally required for ClusteringMetrics

    def global_importance_sampling(self, k: int) -> tuple[dict[str, Tensor], Tensor]:
        raise NotImplementedError  # optionally required for ModelMetrics

    def log_pz(
        self, z: Tensor, global_samples: dict[str, Tensor] | None = None
    ) -> Tensor:
        raise NotImplementedError  # optionally required for ModelMetrics
