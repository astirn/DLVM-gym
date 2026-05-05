import torch
import torch.distributions as dist
import torch.nn.functional as F
from sklearn.metrics.cluster import normalized_mutual_info_score
from torchmetrics import Metric


class ClusteringMetrics(Metric):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.add_state("assignments", default=torch.zeros(self.k))
        self.add_state("accuracy", default=torch.tensor(0.0))
        self.add_state("nmi", default=torch.tensor(0.0))
        self.add_state("num_batches", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))

    def update(
        self, *, x: torch.Tensor, c: torch.Tensor, qc_z: dist.Categorical, **kwargs
    ) -> None:
        cluster_assignments = torch.argmax(qc_z.probs, dim=1)
        cluster_labels = c[torch.argmax(qc_z.probs, dim=0)]
        self.assignments += F.one_hot(cluster_assignments, self.k).sum(dim=0)
        self.accuracy += (c == cluster_labels[cluster_assignments]).sum()
        self.nmi += normalized_mutual_info_score(c.cpu(), cluster_assignments.cpu())
        self.num_batches += 1
        self.num_samples += c.shape[0]

    def compute(self) -> dict[str, torch.Tensor]:
        return {
            "clusters": (getattr(self, "assignments") > 0).sum().item(),
            "acc": getattr(self, "accuracy") / getattr(self, "num_samples"),
            "nmi": getattr(self, "nmi") / getattr(self, "num_batches"),
        }
