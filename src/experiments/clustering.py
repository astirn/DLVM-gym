import hydra
import torch
import torch.distributions as dist
import torch.nn as nn
from omegaconf import DictConfig

from src.train import train


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
        self.log_sigma = nn.Linear(2000, dim_z)

    def forward(self, x: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = self.log_sigma(out).exp()
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
        self.log_sigma = nn.Parameter(torch.as_tensor(0.0))

    def forward(self, z: torch.Tensor) -> dist.Distribution:
        scale = self.log_sigma.exp()
        return dist.Independent(dist.Normal(self.mu(z), scale), 1)


@hydra.main(version_base=None, config_path="../../conf")
def main(cfg: DictConfig) -> None:
    datamodule, model, trainer = train(cfg)
    trainer.test(model, datamodule.test_dataloader())
    model.test_metrics = list(map(instantiate, cfg.metrics.full_metrics))
    trainer.test(model, datamodule.full_dataloader())


if __name__ == "__main__":
    main()
