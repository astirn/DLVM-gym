import torch
import torch.distributions as dist
import torch.nn.functional as F
from sklearn.metrics.cluster import normalized_mutual_info_score
from torchmetrics import Metric


class ClusteringMetrics(Metric):
    def __init__(self, k: int, label_key: str, props_key: str):
        super().__init__()
        self.k = k
        self.label_key = label_key
        self.props_key = props_key
        self.add_state("assignments", default=torch.zeros(self.k))
        self.add_state("accuracy", default=torch.tensor(0.0))
        self.add_state("nmi", default=torch.tensor(0.0))
        self.add_state("num_batches", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))

    def update(
        self,
        batch: dict[str, torch.Tensor],
        distributions: dict[str, dist.Distribution],
    ) -> None:
        if self.label_key in batch and self.props_key in distributions:
            labels = batch[self.label_key]
            qc = distributions[self.props_key]
            if isinstance(qc, dist.Categorical):
                cluster_probs = qc.probs
                cluster_assignments = torch.argmax(cluster_probs, dim=1)
                cluster_labels = labels[torch.argmax(cluster_probs, dim=0)]
                self.assignments += F.one_hot(cluster_assignments, self.k).sum(dim=0)
                self.accuracy += (labels == cluster_labels[cluster_assignments]).sum()
                self.nmi += normalized_mutual_info_score(
                    labels.cpu(), cluster_assignments.cpu()
                )
                self.num_batches += 1
                self.num_samples += labels.shape[0]

    def compute(self) -> dict[str, torch.Tensor]:
        return {
            "clusters": (getattr(self, "assignments") > 0).sum().item(),
            "acc": getattr(self, "accuracy") / getattr(self, "num_samples"),
            "nmi": getattr(self, "nmi") / getattr(self, "num_batches"),
        }
