import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
from torchmetrics import Metric

from src.models.abstract import DeepLatentVariableModel


class ClusteringMetrics(Metric):
    def __init__(self, label_key: str):
        super().__init__()
        self.label_key = label_key
        self.add_state("labels", default=torch.tensor([]))
        self.add_state("assignments", default=torch.tensor([]))
        self.add_state("max_responsibilities", default=torch.tensor([]))
        self.add_state("most_probable_labels", default=torch.tensor([]))

    def update(
        self, batch: dict[str, torch.Tensor], model: DeepLatentVariableModel
    ) -> None:
        if self.label_key in batch:
            labels = batch[self.label_key]

            qz = model.qz(**batch)
            cluster_responsibilities = model.prior.cluster_responsibilities(z=qz.loc)
            cluster_assignments = torch.argmax(cluster_responsibilities.probs, dim=1)
            max_responsibilities = cluster_responsibilities.probs.max(dim=0).values
            cluster_labels = labels[torch.argmax(cluster_responsibilities.probs, dim=0)]

            self.labels = torch.cat([self.labels, labels])
            self.assignments = torch.cat([self.assignments, cluster_assignments])

            if self.max_responsibilities.shape[0] == 0:
                self.max_responsibilities = max_responsibilities
                self.most_probable_labels = cluster_labels
            else:
                replace = max_responsibilities > self.max_responsibilities
                self.max_responsibilities[replace] = max_responsibilities[replace]
                self.most_probable_labels[replace] = cluster_labels[replace]

    def compute(self) -> dict[str, torch.Tensor]:

        if self.assignments.numel() == 0:
            return {
                "clusters": torch.tensor(0.0),
                "acc": torch.tensor(0.0),
                "nmi": torch.tensor(0.0),
            }

        clusters = torch.tensor(self.assignments.unique().numel())
        acc = (
            (self.labels == self.most_probable_labels[self.assignments.int()])
            .float()
            .mean()
        )
        nmi = torch.tensor(
            normalized_mutual_info_score(self.labels.cpu(), self.assignments.cpu())
        )

        return {"clusters": clusters, "acc": acc, "nmi": nmi}
