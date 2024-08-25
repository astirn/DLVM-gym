import argparse
import lightning
import lightning.pytorch.callbacks as callbacks
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.datasets as datasets
import utils_torch as utils

from cytoolz import merge_with
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from typing import Literal


class ClusteringMetrics(torchmetrics.Metric):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.add_state('assignments', default=torch.zeros(self.k))
        self.add_state('accuracy', default=torch.tensor(0.))
        self.add_state('nmi', default=torch.tensor(0.))
        self.add_state('num_batches', default=torch.tensor(0.))
        self.add_state('num_samples', default=torch.tensor(0.))

    def update(self, x: torch.Tensor, y: torch.Tensor, qc_z: dist.Categorical, **kwargs) -> None:
        y_hat = torch.argmax(qc_z.probs, dim=1)
        cluster_labels = y[torch.argmax(qc_z.probs, dim=0)]
        self.assignments += F.one_hot(y_hat, self.k).sum(dim=0)
        self.accuracy += (y == cluster_labels[y_hat]).sum()
        self.nmi += normalized_mutual_info_score(y.cpu(), y_hat.cpu())
        self.num_batches += 1
        self.num_samples += y.shape[0]

    def compute(self) -> dict[str, torch.Tensor]:
        return {
            'clusters': (self.assignments > 0).sum(),
            'acc': self.accuracy / self.num_samples,
            'nmi': self.nmi / self.num_batches,
        }


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

    def forward(self, x: torch.Tensor) -> dist.MultivariateNormal:
        out = self.net(x)
        mu = self.mu(out)
        sigma = self.sigma(out)
        return dist.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))


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

    def forward(self, x: torch.Tensor) -> dist.Distribution:
        return dist.Independent(dist.Normal(self.mu(x), F.softplus(self.log_sigma)), 1)


class VampPrior(lightning.LightningModule):

    def __init__(
            self,
            dim_x: int,
            dim_z: int,
            encoder: callable,
            decoder: callable,
            u: torch.Tensor,
            learning_rate: float,
            prior_mixture: Literal['fixed', 'learnable', 'variational'] = 'fixed',
            prior_variance: Literal['amortized', 'decoupled'] = 'amortized',
    ):
        super().__init__()

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.k = nn.Parameter(torch.as_tensor(u.shape[0]), requires_grad=False)

        self.qz_x = encoder(self.dim_x, self.dim_z)
        self.px_z = decoder(self.dim_x, self.dim_z)

        self.prior_mixture = prior_mixture
        self.prior_variance = prior_variance

        # trainable prior parameters
        self.u = nn.Parameter(u, requires_grad=True)
        self.alpha_uc = nn.Parameter(
            utils.softplus_inverse(torch.ones(self.k)),
            requires_grad=self.prior_mixture == 'variational'
        )
        self.pi_uc = nn.Parameter(
            torch.ones(self.k),
            requires_grad=self.prior_mixture == 'learnable'
        )
        self.L_uc = nn.Parameter(
            torch.eye(self.dim_z)[None].repeat(int(self.k), 1, 1),
            requires_grad=self.prior_variance != 'coupled'
        )

        self.clustering_metrics = ClusteringMetrics(k=self.k)

        self.learning_rate = learning_rate

    def log_pz_c(self, z: torch.Tensor) -> torch.Tensor:
        z = z[:, None]
        pz_c = self.qz_x(self.u)
        if self.prior_variance == 'amortized':
            return pz_c.log_prob(z)
        elif self.prior_variance == 'decoupled':
            return dist.MultivariateNormal(pz_c.mean, scale_tril=utils.scale_tril(self.L_uc)).log_prob(z)
        else:
            raise NotImplementedError

    def qc_z(self, log_pz_c: torch.Tensor) -> dist.Categorical:
        log_probs = log_pz_c + torch.log_softmax(self.pi_uc, dim=0)
        return dist.Categorical(probs=F.softmax(log_probs, dim=1))

    def qpi(self) -> dist.Dirichlet:
        return dist.Dirichlet(F.softplus(self.alpha_uc))

    def dkl_z(self, qz_x: dist.Distribution, log_pz_c: torch.Tensor) -> torch.Tensor:
        if self.prior_mixture == 'variational':
            log_pi = torch.log(self.qpi().rsample())
        else:
            log_pi = torch.log_softmax(self.pi_uc, dim=0)
        return -qz_x.entropy() - torch.logsumexp(log_pi + log_pz_c, dim=-1)

    def dkl_pi(self):
        if self.prior_mixture == 'variational':
            return dist.kl_divergence(self.qpi(), dist.Dirichlet(torch.ones_like(self.alpha_uc) / self.k))
        else:
            return 0

    def evidence_lower_bound(self, x: torch.Tensor) -> (torch.Tensor, dict[str, dist.Distribution]):
        qz_x = self.qz_x(x)
        z = qz_x.rsample()

        log_pz_c = self.log_pz_c(z)

        elbo = (
            self.px_z(z).log_prob(x).sum()
            - self.dkl_z(qz_x, log_pz_c).sum()
            - self.dkl_pi()
        ) / x.shape[0]

        return elbo, {'qz_x': qz_x, 'qc_z': self.qc_z(log_pz_c)}

    def training_step(self, batch, batch_idx):
        elbo, _ = self.evidence_lower_bound(batch['x'])
        self.log("train_elbo", elbo, prog_bar=True)
        return elbo

    def validation_step(self, batch, batch_idx):
        elbo, variational_family = self.evidence_lower_bound(batch['x'])
        self.log("val_elbo", elbo, on_epoch=True, prog_bar=True, batch_size=batch['x'].shape[0])
        self.clustering_metrics.update(**{**batch, **variational_family})

    def on_validation_epoch_end(self):
        self.log_dict(self.clustering_metrics.compute(), on_epoch=True, prog_bar=True)
        self.clustering_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, maximize=True)
        return optimizer


def collate(batch):
    batch = list(map(lambda b: {'x': b[0], 'y': torch.LongTensor([b[1]])}, batch))
    batch = merge_with(torch.vstack, batch)
    batch['x'] = (torch.flatten(batch['x'], start_dim=1) - 0.5) * 2
    batch['y'] = batch['y'].squeeze()
    return batch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='MNIST')
    parser.add_argument('--latent_dim', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--max_clusters', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=1_000)
    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--prior_mixture', type=str, default='fixed')
    parser.add_argument('--prior_variance', type=str, default='amortized')

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
        'drop_last': True,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': True
    }
    dl_train = torch.utils.data.DataLoader(ds_train, shuffle=True, **dl_kwargs)
    dl_test = torch.utils.data.DataLoader(ds_test, shuffle=False, **dl_kwargs)

    # build model
    data_dim = np.prod(ds_train[0][0].shape[1:])
    model = VampPrior(
        dim_x=data_dim,
        dim_z=args.latent_dim,
        encoder=Encoder,
        decoder=Decoder,
        u=next(iter(dl_train))['x'][:args.max_clusters],
        learning_rate=args.learning_rate,
        prior_mixture=args.prior_mixture,
        prior_variance=args.prior_variance
    )

    # train model
    trainer = lightning.Trainer(
        callbacks=[callbacks.EarlyStopping(monitor='val_elbo', patience=20, mode='max')],
        devices=1,
        enable_checkpointing=False,
        logger=False,
        max_epochs=args.max_epochs,
    )
    trainer.fit(model, dl_train, dl_test)
