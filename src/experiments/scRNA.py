import hydra
import lightning
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import DictConfig
from toolz import concat, valmap

from src.data.scRNA import scRNADataModule
from src.model_wrapper import ModelWrapper


class NegativeBinomial(dist.Distribution):
    def __init__(self, mu: torch.Tensor, theta: torch.Tensor, eps: float = 1e-8):
        super().__init__(validate_args=False)
        self.mu = mu
        self.theta = theta
        self.eps = eps

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_p = (self.mu + self.eps).log() - (self.mu + self.theta + self.eps).log()
        log_1_minus_p = (self.theta + self.eps).log() - (
            self.mu + self.theta + self.eps
        ).log()
        log_prob = (
            +x * log_p
            + self.theta * log_1_minus_p
            + (x + self.theta).lgamma()
            - self.theta.lgamma()
            - (x + 1).lgamma()
        )
        return log_prob.sum(dim=1)


class ZeroInflatedNegativeBinomial(dist.Distribution):
    def __init__(
        self,
        mu: torch.Tensor,
        theta: torch.Tensor,
        rho: torch.Tensor,
        eps: float = 1e-8,
    ):
        super().__init__(validate_args=False)
        self.mu = mu
        self.theta = theta
        self.rho = rho
        self.eps = eps

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_p = (self.mu + self.eps).log() - (self.mu + self.theta + self.eps).log()
        log_1_minus_p = (self.theta + self.eps).log() - (
            self.mu + self.theta + self.eps
        ).log()
        zero_ll = F.softplus(-self.rho + self.theta * log_1_minus_p) - F.softplus(
            -self.rho
        )
        non_zero_ll = (
            -self.rho
            - F.softplus(-self.rho)
            + self.theta * log_1_minus_p
            + x * log_p
            + (x + self.theta).lgamma()
            - self.theta.lgamma()
            - (x + 1).lgamma()
        )
        log_prob = torch.where(torch.le(x, 1e-8), zero_ll, non_zero_ll)
        return log_prob.sum(dim=1)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        dim_z: int,
        dim_hidden: int,
        num_batches: int,
        num_genes: int,
        num_layers: int,
        **kwargs,
    ):
        super().__init__()
        dims = [num_genes + num_batches] + [dim_hidden] * num_layers
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
        self.mu = nn.Linear(dims[-1], dim_z)
        self.log_sigma = nn.Linear(dims[-1], dim_z)

    def forward(self, xs: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(xs)
        mu = self.mu(out)
        sigma = (self.log_sigma(out).exp() + 1e-4).sqrt()
        return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        dim_z: int,
        dim_hidden: int,
        gene_likelihood: str,
        num_batches: int,
        num_genes: int,
        num_layers: int,
        **kwargs,
    ):
        super().__init__()
        dims = [dim_z + num_batches] + [dim_hidden] * num_layers
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
        self.log_theta = nn.Parameter(torch.zeros(num_genes))
        self.fw = nn.Sequential(
            nn.Linear(dims[-1], num_genes),
            nn.Softmax(dim=1),
        )
        self.fh = nn.Linear(dims[-1], num_genes)

    def forward(self, zs: torch.Tensor, library_size: torch.Tensor):
        out = self.net(zs)
        mu = self.fw(out) * library_size
        theta = self.log_theta.exp()
        if self.gene_likelihood == "nb":
            return NegativeBinomial(mu, theta)
        elif self.gene_likelihood == "zinb":
            rho = self.fh(out)
            return ZeroInflatedNegativeBinomial(mu, theta, rho)
        else:
            raise NotImplementedError


def train(cfg: DictConfig) -> tuple[scRNADataModule, ModelWrapper, Trainer]:
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    datamodule: scRNADataModule = instantiate(cfg.data)
    datamodule.prepare_data()
    datamodule.setup()

    model = ModelWrapper(
        model=instantiate(
            cfg.model,
            num_batches=len(datamodule.batch_keys),
            num_data=datamodule.num_cells,
            num_genes=datamodule.num_genes,
        ),
        validation_metrics=list(map(instantiate, cfg.metrics.validation_metrics)),
        test_metrics=list(map(instantiate, cfg.metrics.test_metrics)),
        optimizer_params=dict(cfg.optimizer),
        scheduler_params=(
            dict(cfg.scheduler) if cfg.get("scheduler") is not None else None
        ),
    )

    callbacks = list(valmap(instantiate, cfg.callbacks).values())
    trainer = instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(model=model, datamodule=datamodule)

    return datamodule, model, trainer


@hydra.main(version_base=None, config_path="../../conf")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
