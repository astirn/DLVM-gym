import torch
import torch.nn.functional as F
from sklearn.metrics.cluster import normalized_mutual_info_score
from torchmetrics import Metric

from src.models.abstract import DeepLatentVariableModel


class ClusteringMetrics(Metric):
    def __init__(self, label_key: str):
        super().__init__()
        self.label_key = label_key
        self.add_state("assignments", default=torch.zeros(2048))
        self.add_state("accuracy", default=torch.tensor(0.0))
        self.add_state("nmi", default=torch.tensor(0.0))
        self.add_state("num_batches", default=torch.tensor(0.0))
        self.add_state("num_samples", default=torch.tensor(0.0))

    def update(
        self, batch: dict[str, torch.Tensor], model: DeepLatentVariableModel
    ) -> None:
        if self.label_key in batch:
            labels = batch[self.label_key]
            qz = model.qz(**batch)
            component_dist = model.prior.component_distribution(z=qz.sample())
            cluster_assignments = torch.argmax(component_dist.probs, dim=1)
            cluster_labels = labels[torch.argmax(component_dist.probs, dim=0)]
            self.assignments += F.one_hot(
                cluster_assignments, len(self.assignments)
            ).sum(dim=0)
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
