import argparse
import numpy as np
import lightning
import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets

from cytoolz import merge_with, pipe
from cytoolz.curried import map
from torch.fx.experimental.unification.unification_tools import merge_with
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class Encoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_x, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
        )
        self.mu = nn.Linear(2000, dim_z)
        self.sigma = nn.Sequential(
            nn.Linear(2000, dim_z),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> distributions.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = self.sigma(out)
        return distributions.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))

class Decoder(nn.Module):
    def __init__(self, dim_x: int, dim_z: int):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(dim_z, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, dim_x)
        )
        self.log_sigma = nn.Parameter(torch.zeros(dim_x))

    def forward(self, x: torch.Tensor) -> distributions.Distribution:
        return distributions.Independent(distributions.Normal(self.mu(x), F.softplus(self.log_sigma)), 1)

class VampPrior(lightning.LightningModule):

    def __init__(
            self,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            u: torch.Tensor,
            kl_warmup_steps: int,
            learning_rate: float,
    ):
        super().__init__()

        self.qz_x = encoder
        self.px_z = decoder

        self.u = nn.Parameter(u)
        self.k = nn.Parameter(torch.as_tensor(self.u.shape[0]), requires_grad=False)

        self.kl_warmup_step = 0
        self.kl_warmup_steps_max = kl_warmup_steps
        self.learning_rate = learning_rate

    def log_pz_c(self, z: torch.Tensor) -> torch.Tensor:
        return self.qz_x(self.u).log_prob(z[:, None])

    def dkl_z(self, qz_x, log_pz_c):
        dkl_z = -qz_x.entropy() - torch.logsumexp(log_pz_c - torch.log(self.k), dim=-1)
        return dkl_z.sum()

    def evidence_lower_bound(self, x: torch.Tensor) -> torch.Tensor:
        qz_x = self.qz_x(x)
        z = qz_x.sample()

        log_pz_c = self.log_pz_c(z)

        kl_factor = min(1., self.kl_warmup_step / self.kl_warmup_steps_max)
        elbo = (
            self.px_z(z).log_prob(x).sum() -
            self.dkl_z(qz_x, log_pz_c).sum() * kl_factor
        ) / x.shape[0]

        return elbo

    def training_step(self, batch, batch_idx):
        elbo = self.evidence_lower_bound(batch['x'])
        self.kl_warmup_step = min(self.kl_warmup_step + 1, self.kl_warmup_steps_max)
        self.log("train_elbo", elbo, prog_bar=True)
        return elbo

    def validation_step(self, batch, batch_idx):
        elbo = self.evidence_lower_bound(batch['x'])
        self.log("val_elbo", elbo, on_epoch=True, prog_bar=True, batch_size=batch['x'].shape[0])


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, maximize=True)
        return optimizer

def collate(batch):
    batch = pipe(
        batch,
        map(lambda b: {'x': b[0], 'y': torch.LongTensor([b[1]])}),
        list
    )
    batch = merge_with(torch.vstack, batch)
    batch['x'] = (torch.flatten(batch['x'], start_dim=1) - 0.5) * 2
    batch['y'] = batch['y'].squeeze()
    return batch

if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--kl_warmup_steps', type=int, default=5_000)
    parser.add_argument('--latent_dim', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_clusters', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    assert args.batch_size > args.max_clusters, 'pseudo-inputs initialization makes this assumption'

    # random seed
    if args.seed is not None:
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)

    # load dataset
    dataset = getattr(datasets, args.dataset)
    ds_train = dataset(root='data', train=True, download=True, transform=ToTensor())
    ds_test = dataset(root='data', train=False, download=True, transform=ToTensor())

    # create dataloaders
    dl_kwargs = {
        'batch_size': args.batch_size,
        'collate_fn': collate,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': True
    }
    dl_train = torch.utils.data.DataLoader(ds_train, shuffle=True, **dl_kwargs)
    dl_test = torch.utils.data.DataLoader(ds_test, shuffle=False, **dl_kwargs)

    # build model
    data_dim = np.prod(ds_train[0][0].shape[1:])
    model = VampPrior(
        encoder=Encoder(dim_x=data_dim, dim_z=args.latent_dim),
        decoder=Decoder(dim_x=data_dim, dim_z=args.latent_dim),
        u=next(iter(dl_train))['x'][:args.max_clusters],
        kl_warmup_steps=args.kl_warmup_steps,
        learning_rate=args.learning_rate,
    )

    # train model
    trainer = lightning.Trainer(max_epochs=args.max_epochs, devices=1)
    trainer.fit(model, dl_train, dl_test)
