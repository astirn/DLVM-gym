import hydra
import lightning
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import DictConfig
from toolz import concat, merge_with, valmap

from src.data.scRNA import scRNADataModule
from src.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from src.metrics.scib_metrics import scib_metrics
from src.model_wrapper import ModelWrapper
from src.utils import concat_distributions


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

    checkpoint = torch.load(
        trainer.checkpoint_callback.best_model_path, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])

    return datamodule, model, trainer


def analyis(
    cfg: DictConfig, datamodule: scRNADataModule, model: ModelWrapper, trainer: Trainer
) -> None:
    distributions = trainer.predict(model, datamodule.predict_dataloader())
    distributions = merge_with(concat_distributions, distributions)
    z = distributions["qz_xs"].sample()
    adata = datamodule.adata
    adata.obsm["scVI"] = z.numpy()
    df = scib_metrics(adata, ["scVI"])


@hydra.main(version_base=None, config_path="../../conf")
def main(cfg: DictConfig) -> None:
    datamodule, model, trainer = train(cfg)
    analyis(cfg, datamodule, model, trainer)


if __name__ == "__main__":
    main()
