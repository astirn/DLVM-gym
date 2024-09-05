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
            'clusters': (self.assignments > 0).sum().to(torch.float32),
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
            num_data: int,
            encoder: callable,
            decoder: callable,
            learning_rate: float,
            truncation_level: int,
            u_initial: torch.Tensor = None,
            u_constraint: callable = None,
            categorical_prior: bool = False,
            dirichlet_prior: bool = False,
            dirichlet_process: bool = False,
            inverse_gamma_prior: bool = False,
            wishart_prior: bool = False,
    ):
        assert sum([categorical_prior, dirichlet_prior, dirichlet_process]) <= 1
        assert not inverse_gamma_prior or not dirichlet_prior or not dirichlet_process
        super().__init__()

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.num_data = num_data

        self.qz_x = encoder(self.dim_x, self.dim_z)
        self.px_z = decoder(self.dim_x, self.dim_z)

        self.k = nn.Parameter(torch.as_tensor(truncation_level), requires_grad=False)
        self.u_constraint = (lambda x: x) if u_constraint is None else u_constraint
        self.u_uc = nn.Parameter(
            torch.randn([self.k, self.dim_x]) if u_initial is None else u_initial,
            requires_grad=True
        )

        self.categorical_prior = categorical_prior
        self.dirichlet_prior = dirichlet_prior
        self.dirichlet_process = dirichlet_process
        self.inverse_gamma_prior = inverse_gamma_prior
        self.wishart_prior = wishart_prior

        # global variational parameters (unconstrained)
        self.a_uc = nn.Parameter(
            utils.softplus_inverse(torch.ones(1)),
            requires_grad=self.inverse_gamma_prior
        )
        self.b_uc = nn.Parameter(
            utils.softplus_inverse(torch.ones(1)),
            requires_grad=self.inverse_gamma_prior
        )
        self.alpha_uc = nn.Parameter(
            utils.softplus_inverse(torch.ones(self.k) * 2),
            requires_grad=self.dirichlet_prior or self.dirichlet_process
        )
        self.beta_uc = nn.Parameter(
            utils.softplus_inverse(torch.ones(self.k) * self.k),
            requires_grad=self.dirichlet_process
        )
        self.n_uc = nn.Parameter(
            utils.softplus_inverse(self.dim_z * torch.ones(self.k) + 2),
            requires_grad=self.wishart_prior
        )
        self.L_uc = nn.Parameter(
            torch.eye(self.dim_z)[None].repeat(self.k, 1, 1),
            requires_grad=self.wishart_prior
        )

        self.clustering_metrics = ClusteringMetrics(k=self.k)

        self.learning_rate = learning_rate

    def a_and_b(self):
        return F.softplus(self.a_uc), F.softplus(self.b_uc)

    def alpha_and_beta(self):
        return F.softplus(self.alpha_uc), F.softplus(self.beta_uc)

    def n_and_V(self):
        n = F.softplus(self.n_uc)
        L = utils.scale_tril(self.L_uc)
        return n, L @ L.mT

    def log_pz_c(self, z: torch.Tensor) -> torch.Tensor:
        z = z[:, None]
        pz_c = self.qz_x(self.u_constraint(self.u_uc))
        if self.wishart_prior:
            n, V = self.n_and_V()
            log_det = (
                    utils.multivariate_digamma(n / 2, self.dim_z) +
                    self.dim_z * torch.log(torch.as_tensor(2)) +
                    torch.linalg.slogdet(V)[1]
            )
            residuals = z - pz_c.mean
            quadratic = torch.squeeze(
                torch.unsqueeze(residuals, dim=2)
                @ (n[:, None, None] * V) @
                torch.unsqueeze(residuals, dim=3)
            )
            return 0.5 * (-self.dim_z * torch.log(torch.as_tensor(2 * torch.pi)) + log_det - quadratic)
        else:
            return pz_c.log_prob(z)

    def log_v_and_log_vm1(self) -> (torch.Tensor, torch.Tensor):
        alpha, beta = self.alpha_and_beta()
        log_v = (torch.digamma(alpha) - torch.digamma(alpha + beta))[:-1]
        log_v = torch.concat([log_v, torch.as_tensor([0]).to(log_v.device)])
        log_vm1 = (torch.digamma(beta) - torch.digamma(alpha + beta))[:-1]
        log_vm1 = torch.concat([log_vm1, torch.as_tensor([-torch.inf]).to(log_vm1.device)])
        return log_v, log_vm1

    def qc_z(self, log_pz_c: torch.Tensor) -> dist.Categorical:
        log_probs = log_pz_c
        if self.categorical_prior:
            log_probs -= torch.log(self.k)
        elif self.dirichlet_prior:
            alpha, _ = self.alpha_and_beta()
            log_probs += torch.digamma(alpha) - torch.digamma(alpha.sum())
        elif self.dirichlet_process:
            log_v, log_vm1 = self.log_v_and_log_vm1()
            log_probs += log_v + torch.concat([torch.zeros_like(log_vm1)[:1], torch.cumsum(log_vm1, dim=0)[:-1]])

        return dist.Categorical(probs=F.softmax(log_probs, dim=1))

    def qalpha0(self):
        return dist.InverseGamma(*self.a_and_b())

    def dkl_z(self, qz_x, qc_z, log_pz_c) -> torch.Tensor:
        dkl_z = -qz_x.entropy()
        if self.categorical_prior or self.dirichlet_prior or self.dirichlet_process:
            dkl_z -= (qc_z.probs * log_pz_c).sum(dim=-1)
        else:
            dkl_z -= torch.logsumexp(log_pz_c - torch.log(self.k), dim=-1)
        return dkl_z.sum()

    def dkl_c(self, qc_z: dist.Categorical):
        if self.categorical_prior:
            pc = dist.Categorical(probs=torch.ones(self.k, device=self.k.device) / self.k)
            return dist.kl_divergence(qc_z, pc).sum()
        elif self.dirichlet_prior:
            alpha, _ = self.alpha_and_beta()
            dkl_c = -qc_z.entropy() - (qc_z.probs * (torch.digamma(alpha) - torch.digamma(alpha.sum()))).sum(dim=1)
            return dkl_c.sum()
        elif self.dirichlet_process:
            log_v, log_vm1 = self.log_v_and_log_vm1()
            qc_gtj_z = torch.fliplr(torch.cumsum(torch.fliplr(qc_z.probs), dim=1)[:, :-1])
            qc_gtj_z = torch.concat([qc_gtj_z, torch.zeros_like(qc_gtj_z)[:, :1]], dim=1)
            dkl_c = -qc_z.entropy() - (qc_z.probs * log_v).sum(dim=1) - (qc_gtj_z[:, :-1] * log_vm1[:-1]).sum(dim=1)
            return dkl_c.sum()
        else:
            return 0

    def dkl_pi(self):
        alpha, _ = self.alpha_and_beta()
        q = dist.Dirichlet(alpha)
        if self.dirichlet_prior and self.inverse_gamma_prior:
            alpha0 = self.qalpha0().rsample()  # TODO: anything more than a single MC sample loses gradients
            p = dist.Dirichlet(alpha0 * torch.ones_like(alpha) / self.k)
            return dist.kl_divergence(q, p)

        elif self.dirichlet_prior:
            p = dist.Dirichlet(torch.ones_like(alpha) / self.k)
            return dist.kl_divergence(q, p)
        else:
            return 0

    def dkl_v(self):
        q = dist.Beta(*self.alpha_and_beta())
        if self.dirichlet_process and self.inverse_gamma_prior:
            alpha0 = self.qalpha0().rsample()  # TODO: anything more than a single MC sample loses gradients
            return dist.kl_divergence(q, dist.Beta(1, alpha0))[:-1].sum()
        elif self.dirichlet_process:
            return dist.kl_divergence(q, dist.Beta(1, 1))[:-1].sum()
        else:
            return 0

    def dkl_alpha0(self):
        if self.dirichlet_prior and self.inverse_gamma_prior:
            return dist.kl_divergence(self.qalpha0(), dist.InverseGamma(1, 1))
        elif self.dirichlet_process and self.inverse_gamma_prior:
            return dist.kl_divergence(self.qalpha0(), dist.InverseGamma(1, 30))
        else:
            return 0

    def dkl_Lambda(self):
        if self.wishart_prior:
            n_q, V_q = self.n_and_V()
            n_p = torch.as_tensor(self.dim_z + 2).to(n_q.device)
            V_p_inv = torch.eye(self.dim_z).to(V_q.device) / self.k ** (2 / self.dim_z) * (self.dim_z + 2)
            V_p_inv_V_q = V_p_inv @ V_q
            dkl_Lambda = (
                    - n_p * torch.slogdet(V_p_inv_V_q).logabsdet
                    + n_q * (torch.einsum('bii->b', V_p_inv_V_q) - self.dim_z)
                    + 2 * torch.mvlgamma(n_p / 2, self.dim_z)
                    - 2 * torch.mvlgamma(n_q / 2, self.dim_z)
                    + (n_q - n_p) * utils.multivariate_digamma(n_q / 2, self.dim_z)
            ) / 2
            return dkl_Lambda.sum()
        else:
            return 0

    def evidence_lower_bound(self, x: torch.Tensor) -> (torch.Tensor, dict[str, dist.Distribution]):
        qz_x = self.qz_x(x)
        z = qz_x.rsample()

        log_pz_c = self.log_pz_c(z)
        qc_z = self.qc_z(log_pz_c)

        elbo_locals = (
            + self.px_z(z).log_prob(x).sum()
            - self.dkl_z(qz_x, qc_z, log_pz_c)
            - self.dkl_c(qc_z)  # zero only for original VampPrior
        ) / x.shape[0]
        elbo_globals = (
            - self.dkl_pi()  # zero unless Dirichlet prior
            - self.dkl_v()  # zero unless Dirichlet process
            - self.dkl_alpha0()  # zero unless Inverse Gamma prior
            - self.dkl_Lambda()  # zero unless Wishart prior
        ) / self.num_data
        elbo = elbo_locals + elbo_globals

        return elbo, {'qz_x': qz_x, 'qc_z': qc_z}

    def training_step(self, batch, batch_idx):
        elbo, _ = self.evidence_lower_bound(batch['x'])
        self.log("train_elbo", elbo, prog_bar=True)
        return elbo

    def validation_step(self, batch, batch_idx):
        elbo, variational_family = self.evidence_lower_bound(batch['x'])
        self.log("val_elbo", elbo, on_epoch=True, prog_bar=True, batch_size=batch['x'].shape[0])
        self.clustering_metrics.update(**{**batch, **variational_family})

    def on_validation_epoch_end(self) -> None:
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
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=1_000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--truncation_level', type=int, default=100)

    parser.add_argument('--categorical_prior', action='store_true', default=False)
    parser.add_argument('--dirichlet_prior', action='store_true', default=False)
    parser.add_argument('--dirichlet_process', action='store_true', default=False)
    parser.add_argument('--inverse_gamma_prior', action='store_true', default=False)
    parser.add_argument('--wishart_prior', action='store_true', default=False)

    args = parser.parse_args()

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
        num_data=len(ds_train),
        encoder=Encoder,
        decoder=Decoder,
        # u_initial=next(iter(dl_train))['x'][:args.truncation_level],
        u_constraint=F.tanh,
        truncation_level=args.truncation_level,
        learning_rate=args.learning_rate,
        categorical_prior=args.categorical_prior,
        dirichlet_prior=args.dirichlet_prior,
        dirichlet_process=args.dirichlet_process,
        inverse_gamma_prior=args.inverse_gamma_prior,
        wishart_prior=args.wishart_prior,
    )

    # train model
    torch.set_float32_matmul_precision('high')
    trainer = lightning.Trainer(
        callbacks=[callbacks.EarlyStopping(monitor='val_elbo', patience=20, mode='max')],
        devices=1,
        enable_checkpointing=False,
        logger=False,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, dl_train, dl_test)
