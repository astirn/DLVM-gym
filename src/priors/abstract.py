from abc import ABC, abstractmethod

from torch import Tensor
from torch.distributions import Categorical, Distribution
from torch.nn import Module


class Prior(Module, ABC):

    @property
    def has_global_latent_variables(self) -> bool:
        return False

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
