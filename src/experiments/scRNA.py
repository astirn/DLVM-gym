import hydra
import torch
import torch.distributions as dist
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from toolz import concat

from src.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from src.fit import fit


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
                            nn.LayerNorm(dims[i + 1]),
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
                            nn.LayerNorm(dims[i + 1]),
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
            nn.Softmax(dim=-1),
        )
        self.fh = nn.Linear(dims[-1], num_genes)

    def forward(self, zs: torch.Tensor, library_size: torch.Tensor):
        out = self.net(zs)
        mu = self.fw(out) * library_size.unsqueeze(1)
        theta = self.log_theta.exp()
        if self.gene_likelihood == "nb":
            return NegativeBinomial(mu, theta)
        elif self.gene_likelihood == "zinb":
            rho = self.fh(out)
            return ZeroInflatedNegativeBinomial(mu, theta, rho)
        else:
            raise NotImplementedError


@hydra.main(version_base=None, config_path="../../conf")
def main(cfg: DictConfig) -> None:
    datamodule, model, trainer = fit(cfg)

    # Log marginal uses best validation loss checkpoint (i.e. best validation ELBO)
    ckpt = cfg.callbacks.checkpoint_loss
    ckpt_path = f"{ckpt.dirpath}/{ckpt.filename}.ckpt"
    trainer.test(model, datamodule.test_dataloader(), ckpt_path=ckpt_path)

    # scIB and NMI on full dataset uses best validation NMI checkpoint
    ckpt = cfg.callbacks.checkpoint_nmi
    ckpt_path = f"{ckpt.dirpath}/{ckpt.filename}.ckpt"
    model.test_metrics = list(map(instantiate, cfg.metrics.full_metrics))
    trainer.test(model, datamodule.full_dataloader(), ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
