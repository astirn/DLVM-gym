import hydra
import torch
import torch.distributions as dist
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.fit import fit
from src.utils import softplus_inverse


class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x, 500),
            nn.ELU(),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Linear(500, 2000),
            nn.ELU(),
        )
        self.mu = nn.Linear(2000, dim_z)
        self.sigma = nn.Sequential(nn.Linear(2000, dim_z), nn.Softplus())

    def forward(self, x: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = self.sigma(out)
        return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))


class Decoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int, **kwargs):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(dim_z, 2000),
            nn.ELU(),
            nn.Linear(2000, 500),
            nn.ELU(),
            nn.Linear(500, 500),
            nn.ELU(),
            nn.Linear(500, dim_x),
        )
        self.log_sigma = nn.Parameter(softplus_inverse(torch.as_tensor(1.0)))

    def forward(self, z: torch.Tensor) -> dist.Distribution:
        scale = nn.functional.softplus(self.log_sigma)
        return dist.Independent(dist.Normal(self.mu(z), scale), 1)


@hydra.main(version_base=None, config_path="../../conf")
def main(cfg: DictConfig) -> None:
    datamodule, model, trainer = fit(cfg)

    # Log marginal uses best validation loss checkpoint (i.e. best validation ELBO)
    ckpt = cfg.callbacks.checkpoint_loss
    ckpt_path = f"{ckpt.dirpath}/{ckpt.filename}.ckpt"
    trainer.test(model, datamodule.test_dataloader(), ckpt_path=ckpt_path)

    # NMI on full dataset uses best validation NMI checkpoint
    ckpt = cfg.callbacks.checkpoint_nmi
    ckpt_path = f"{ckpt.dirpath}/{ckpt.filename}.ckpt"
    model.test_metrics = list(map(instantiate, cfg.metrics.full_metrics))
    trainer.test(model, datamodule.full_dataloader(), ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
