import hydra
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from toolz import concat

from src.data.scRNA import scRNABatch
from src.train import train


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim_x: int,
        dim_s: int,
        dim_z: int,
        n_layers: int,
        n_hidden: int,
        **kwargs,
    ):
        super().__init__()
        dims = [dim_x + dim_s] + [n_hidden] * n_layers
        self.net = nn.Sequential(
            *list(
                concat(
                    [
                        [
                            nn.Linear(dims[i], dims[i + 1]),
                            nn.BatchNorm1d(dims[i + 1], momentum=0.01, eps=0.001),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                        ]
                        for i in range(len(dims) - 1)
                    ]
                )
            )
        )
        self.mu = nn.Linear(n_hidden, dim_z)
        self.log_sigma = nn.Linear(n_hidden, dim_z)

    def forward(self, batch: scRNABatch) -> dist.MultivariateNormal:
        xs = torch.concat([batch.x.log1p(), batch.s], dim=1)
        out = self.net(xs)
        mu = self.mu(out)
        sigma = (self.log_sigma(out).exp() + 1e-4).sqrt()
        return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        dim_in: int,
        dim_out: int,
        gene_likelihood: str,
        n_layers: int,
        n_hidden: int,
        **kwargs,
    ):
        super().__init__()
        dims = [dim_in] + [n_hidden] * n_layers
        self.net = nn.Sequential(
            *list(
                concat(
                    [
                        [
                            nn.Linear(dims[i], dims[i + 1]),
                            nn.BatchNorm1d(dims[i + 1], momentum=0.01, eps=0.001),
                            nn.ReLU(),
                        ]
                        for i in range(len(dims) - 1)
                    ]
                )
            )
        )
        self.gene_likelihood = gene_likelihood
        self.log_theta = nn.Parameter(torch.zeros(dim_out))
        self.fw = nn.Sequential(
            nn.Linear(dims[-1], dim_out),
            nn.Softmax(dim=1),
        )
        self.fh = nn.Linear(dims[-1], dim_out)

    def forward(self, z: torch.Tensor, batch: dict[str, torch.Tensor], eps=1e-8):
        x, s = batch["x"], batch["s"]
        out = self.net(torch.concat([z, s], dim=1))
        mu = self.fw(out) * x.sum(dim=1, keepdim=True)
        theta = self.log_theta.exp()
        log_p = (mu + eps).log() - (mu + theta + eps).log()
        log_1_minus_p = (theta + eps).log() - (mu + theta + eps).log()
        if self.gene_likelihood == "nb":
            log_prob = (
                +x * log_p
                + theta * log_1_minus_p
                + (x + theta).lgamma()
                - theta.lgamma()
                - (x + 1).lgamma()
            )
        elif self.gene_likelihood == "zinb":
            fh = self.fh(out)
            zero_ll = F.softplus(-fh + theta * log_1_minus_p) - F.softplus(-fh)
            non_zero_ll = (
                -fh
                - F.softplus(-fh)
                + theta * log_1_minus_p
                + x * log_p
                + (x + theta).lgamma()
                - theta.lgamma()
                - (x + 1).lgamma()
            )
            log_prob = torch.where(torch.le(x, 1e-8), zero_ll, non_zero_ll)
        else:
            raise NotImplementedError

        return log_prob.sum(dim=1)


@hydra.main(
    version_base=None, config_path="../../conf", config_name="experiments/scRNA/retina"
)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
