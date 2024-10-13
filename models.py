import lightning
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from sklearn.metrics.cluster import normalized_mutual_info_score


def multivariate_digamma(a: torch.Tensor, p: int) -> torch.Tensor:
    return torch.stack([torch.digamma(a + (1 - i) / 2) for i in range(1, p + 1)]).sum(dim=0)


class ClusteringMetrics(torchmetrics.Metric):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.add_state('assignments', default=torch.zeros(self.k))
        self.add_state('accuracy', default=torch.tensor(0.))
        self.add_state('nmi', default=torch.tensor(0.))
        self.add_state('num_batches', default=torch.tensor(0.))
        self.add_state('num_samples', default=torch.tensor(0.))

    def update(self, x: torch.Tensor, c: torch.Tensor, qc_z: dist.Categorical, **kwargs) -> None:
        cluster_assignments = torch.argmax(qc_z.probs, dim=1)
        cluster_labels = c[torch.argmax(qc_z.probs, dim=0)]
        self.assignments += F.one_hot(cluster_assignments, self.k).sum(dim=0)
        self.accuracy += (c == cluster_labels[cluster_assignments]).sum()
        self.nmi += normalized_mutual_info_score(c.cpu(), cluster_assignments.cpu())
        self.num_batches += 1
        self.num_samples += c.shape[0]

    def compute(self) -> dict[str, torch.Tensor]:
        return {
            'clusters': (self.assignments > 0).sum().to(torch.float32),
            'acc': self.accuracy / self.num_samples,
            'nmi': self.nmi / self.num_batches,
        }


class ModelMetrics(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state('log_marginal', default=torch.tensor(0.))
        self.add_state('posterior_predictive', default=torch.tensor(0.))
        self.add_state('num_samples', default=torch.tensor(0.))

    def update(self, log_marginal: torch.Tensor, posterior_predictive: torch.Tensor) -> None:
        self.log_marginal += log_marginal
        self.posterior_predictive += posterior_predictive
        self.num_samples += 1

    def compute(self) -> dict[str, torch.Tensor]:
        return {
            'log_marginal': self.log_marginal / self.num_samples,
            'posterior_predictive': self.posterior_predictive / self.num_samples,
        }


class VampPrior(lightning.LightningModule):

    def __init__(
            self,
            dim_x: int,
            dim_z: int,
            num_data: int,
            encoder: nn.Module,
            decoder: nn.Module,
            learning_rate: float,
            truncation_level: int,
            u_constraint: callable = None,
            u_initial: torch.Tensor = None,
            categorical_prior: bool = False,
            decoupled_variance: bool = False,
            dirichlet_prior: bool = False,
            dirichlet_process: bool = False,
            dirichlet_process_variational_family: bool = False,
            wishart_prior: bool = False,
    ):
        assert sum([categorical_prior, dirichlet_prior, dirichlet_process]) <= 1
        super().__init__()
        self.save_hyperparameters()

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.num_data = num_data

        self.qz_x = encoder
        self.px_z = decoder

        self.k = nn.Parameter(torch.as_tensor(truncation_level), requires_grad=False)
        self.u_constraint = (lambda x: x) if u_constraint is None else u_constraint
        self.u_uc = nn.Parameter(
            torch.randn([self.k, self.dim_x]) if u_initial is None else u_initial,
            requires_grad=True
        )

        self.categorical_prior = categorical_prior
        self.decoupled_variance = decoupled_variance
        self.dirichlet_prior = dirichlet_prior
        self.dirichlet_process = dirichlet_process
        self.wishart_prior = wishart_prior
        self.dirichlet_process_variational_family = dirichlet_process_variational_family

        # variational parameters for cluster proportions
        alpha = torch.ones(self.k)
        self.alpha_uc = nn.Parameter(alpha.log(), requires_grad=self.dirichlet_prior or self.dirichlet_process)
        one_minus_v = 1 - 1 / self.k[None]
        for i in range(1, self.k):
            v = 1 / self.k / torch.prod(one_minus_v)
            one_minus_v = torch.concat([one_minus_v, 1 - v[None]])
        v = 1 - one_minus_v
        v[-1] = 1
        beta = alpha / v - alpha
        beta[-1] = 1
        self.beta_uc = nn.Parameter(beta.log(), requires_grad=self.dirichlet_process)

        # variational parameters for cluster precisions
        self.n_uc = nn.Parameter(torch.zeros(self.k), requires_grad=self.wishart_prior)
        L = torch.eye(self.dim_z)[None].repeat(self.k, 1, 1)
        self.L_uc = nn.Parameter(
            L.tril(-1) + L.diagonal(dim1=-2, dim2=-1).log().diag_embed(),
            requires_grad=self.decoupled_variance or self.wishart_prior,
        )

        self.clustering_metrics = ClusteringMetrics(k=self.k)
        self.model_metrics = ModelMetrics()

        self.learning_rate = learning_rate

    def alpha_and_beta(self):
        alpha = self.alpha_uc.exp()
        if self.dirichlet_process_variational_family:
            beta = self.beta_uc.exp()
        else:
            beta = torch.concat([alpha[1:].cumsum(dim=0), torch.as_tensor([1]).to(alpha.device)])

        return alpha, beta

    def n_and_V(self):
        n = self.n_uc.exp() + self.dim_z
        L = self.L_uc.tril(-1) + self.L_uc.diagonal(dim1=-2, dim2=-1).exp().diag_embed()
        V = L @ L.mT
        if self.wishart_prior:
            V = V / n[:, None, None]
        return n, V

    def pc(self):
        if self.dirichlet_prior or (self.dirichlet_process and not self.dirichlet_process_variational_family):
            alpha, _ = self.alpha_and_beta()
            probs = alpha / alpha.sum()
        elif self.dirichlet_process:
            alpha, beta = self.alpha_and_beta()
            v = (alpha / (alpha + beta))[:-1]
            v = torch.concat([v, torch.as_tensor([1]).to(v.device)])
            vm1 = (beta / (alpha + beta))[:-1]
            vm1 = torch.concat([vm1, torch.as_tensor([0]).to(v.device)])
            probs = v * torch.concat([torch.ones_like(vm1)[:1], torch.cumprod(vm1, dim=0)[:-1]])
        else:
            probs = torch.ones(self.k) / self.k

        return dist.Categorical(probs=probs)

    def pz_c(self, z: torch.Tensor) -> torch.Tensor:
        z = z[:, None]
        pz_c = self.qz_x(self.u_constraint(self.u_uc))
        if self.decoupled_variance:
            return dist.MultivariateNormal(loc=pz_c.loc, precision_matrix=self.n_and_V()[1]).log_prob(z).exp()
        elif self.wishart_prior:
            n, V = self.n_and_V()
            nu = n - self.dim_z + 1
            Lambda = (nu[:, None, None] * V)
            quadratic = torch.squeeze(
                torch.unsqueeze(z - pz_c.mean, dim=2)
                @ Lambda @
                torch.unsqueeze(z - pz_c.mean, dim=3)
            )
            ll = (
                + torch.lgamma((nu + self.dim_z) / 2)
                - torch.lgamma(nu / 2)
                - 0.5 * self.dim_z * (nu * torch.pi).log()
                + 0.5 * Lambda.slogdet().logabsdet
                - 0.5 * (nu + self.dim_z) * (1 + quadratic / nu).log()
            )
            return ll.exp()
        else:
            return pz_c.log_prob(z).exp()

    def log_pz_c(self, z: torch.Tensor) -> torch.Tensor:
        z = z[:, None]
        pz_c = self.qz_x(self.u_constraint(self.u_uc))
        if self.wishart_prior:
            n, V = self.n_and_V()
            log_det = (
                    multivariate_digamma(n / 2, self.dim_z) +
                    self.dim_z * torch.log(torch.as_tensor(2)) +
                    torch.linalg.slogdet(V).logabsdet
            )
            residuals = z - pz_c.mean
            quadratic = torch.squeeze(
                torch.unsqueeze(residuals, dim=2)
                @ (n[:, None, None] * V) @
                torch.unsqueeze(residuals, dim=3)
            )
            return -0.5 * (self.dim_z * torch.log(torch.as_tensor(2 * torch.pi)) - log_det + quadratic)
        elif self.decoupled_variance:
            return dist.MultivariateNormal(loc=pz_c.loc, precision_matrix=self.n_and_V()[1]).log_prob(z)
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
        elif self.dirichlet_prior or (self.dirichlet_process and not self.dirichlet_process_variational_family):
            alpha, _ = self.alpha_and_beta()
            log_probs += torch.digamma(alpha) - torch.digamma(alpha.sum())
        elif self.dirichlet_process:
            log_v, log_vm1 = self.log_v_and_log_vm1()
            log_probs += log_v + torch.concat([torch.zeros_like(log_vm1)[:1], torch.cumsum(log_vm1, dim=0)[:-1]])

        return dist.Categorical(logits=log_probs)

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
        elif self.dirichlet_prior or (self.dirichlet_process and not self.dirichlet_process_variational_family):
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
        if self.dirichlet_prior:
            alpha, _ = self.alpha_and_beta()
            q = dist.Dirichlet(alpha)
            p = dist.Dirichlet(torch.ones_like(alpha) / self.k)
            return dist.kl_divergence(q, p)
        else:
            return 0

    def dkl_v(self):
        if self.dirichlet_process:
            q = dist.Beta(*self.alpha_and_beta())
            p = dist.Beta(1, 1)
            return dist.kl_divergence(q, p)[:-1].sum()
        else:
            return 0

    def dkl_Lambda(self, qz_x):
        if self.wishart_prior:
            n_q, V_q = self.n_and_V()

            n_p = torch.as_tensor(self.dim_z + 2).to(n_q.device)
            with torch.no_grad():
                covariance = torch.cov(qz_x.rsample().T)
            if self.dirichlet_prior or (self.dirichlet_process and not self.dirichlet_process_variational_family):
                covariance_proportions = dist.Dirichlet(self.alpha_and_beta()[0]).rsample()
            elif self.dirichlet_process:
                v = dist.Beta(*self.alpha_and_beta()).rsample()[:-1]
                one = torch.ones_like(v)[:1]
                v = torch.concat([v, one])
                one_minus_v = 1 - v
                covariance_proportions = v * torch.concat([one, torch.cumprod(one_minus_v, dim=0)[:-1]])
            else:
                covariance_proportions = 1 / self.k[None]
            covariance_proportions = torch.clamp_min(covariance_proportions ** (1 / self.dim_z), 0.01)
            V_p_inv = n_p * covariance * covariance_proportions[:, None, None]

            V_p_inv_V_q = V_p_inv @ V_q
            dkl_Lambda = (
                    - n_p * torch.slogdet(V_p_inv_V_q).logabsdet
                    + n_q * (torch.einsum('bii->b', V_p_inv_V_q) - self.dim_z)
                    + 2 * torch.mvlgamma(n_p / 2, self.dim_z)
                    - 2 * torch.mvlgamma(n_q / 2, self.dim_z)
                    + (n_q - n_p) * multivariate_digamma(n_q / 2, self.dim_z)
            ) / 2
            return dkl_Lambda.sum()
        else:
            return 0

    def evidence_lower_bound(self, batch: dict[str, torch.Tensor]) -> (torch.Tensor, dict[str, dist.Distribution]):
        x = batch['x']
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
            - self.dkl_Lambda(qz_x)  # zero unless Wishart prior
        ) / self.num_data
        elbo = elbo_locals + elbo_globals

        return elbo, {'qz_x': qz_x, 'qc_z': qc_z}

    def training_step(self, batch, batch_idx):
        elbo, _ = self.evidence_lower_bound(batch)
        self.log("train_elbo", elbo, on_epoch=True, on_step=False, prog_bar=True)
        return elbo

    def validation_step(self, batch, batch_idx):
        elbo, variational_family = self.evidence_lower_bound(batch)
        self.log("val_elbo", elbo, on_epoch=True, prog_bar=True, batch_size=batch['x'].shape[0])
        self.clustering_metrics.update(**{**batch, **variational_family})

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.clustering_metrics.compute(), on_epoch=True, prog_bar=True)
        self.clustering_metrics.reset()

    def test_step(self, batch, batch_idx):
        elbo = [self.evidence_lower_bound(batch)[0] for _ in range(1_000)]
        log_marginal = torch.logsumexp(torch.stack(elbo), dim=0) - torch.tensor(1_000).float().log()
        log_marginal = log_marginal.mean()
        log_likelihood = [self.px_z(self.qz_x(batch['x']).rsample()).log_prob(batch['x']) for _ in range(1_000)]
        posterior_predictive = torch.logsumexp(torch.stack(log_likelihood), dim=0) - torch.tensor(1_000).float().log()
        posterior_predictive = posterior_predictive.mean()
        self.model_metrics.update(log_marginal, posterior_predictive)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.model_metrics.compute(), on_epoch=True, prog_bar=True)
        self.model_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            maximize=True,
            weight_decay=0,
        )

        return optimizer


class SingleCellVampPrior(VampPrior):
    def __init__(
            self,
            n_cells: int,
            n_genes: int,
            n_batches: int,
            n_latent: int,
            encoder: callable,
            decoder: callable,
            truncation_level: int,
            learning_rate: float,
            **kwargs,
    ):
        VampPrior.__init__(
            self,
            dim_x=n_genes + n_batches,  # only genes are modelled, but this sets the pseudo-input dimension
            dim_z=n_latent,
            num_data=n_cells,
            u_constraint=self.u_constraint,
            encoder=encoder,
            decoder=decoder,
            truncation_level=truncation_level,
            learning_rate=learning_rate,
            **kwargs
        )
        self.save_hyperparameters()

        self.n_genes = n_genes
        self.log_px_zs = self.px_z  # renamed since decoder returns a log probability rather a distribution

    def u_constraint(self, u_uc: torch.Tensor) -> torch.Tensor:
        return torch.concat([
            u_uc[:, :self.n_genes],
            F.softmax(u_uc[:, self.n_genes:], dim=1)
        ], dim=1)

    def evidence_lower_bound(self, batch: dict[str, torch.Tensor]) -> (torch.Tensor, dict[str, dist.Distribution]):
        x, s = batch['x'], batch['s']
        qz_x = self.qz_x(torch.concat([x.log1p(), s], dim=1))
        z = qz_x.rsample()

        log_pz_c = self.log_pz_c(z)
        qc_z = self.qc_z(log_pz_c)

        elbo_locals = (
            + self.log_px_zs(x, z, s).sum()
            - self.dkl_z(qz_x, qc_z, log_pz_c)
            - self.dkl_c(qc_z)  # zero only for original VampPrior
        ) / x.shape[0]
        elbo_globals = (
            - self.dkl_pi()  # zero unless Dirichlet prior
            - self.dkl_v()  # zero unless Dirichlet process
            - self.dkl_Lambda(qz_x)  # zero unless Wishart prior
        ) / self.num_data
        elbo = elbo_locals + elbo_globals

        return elbo, {'qz_x': qz_x, 'qc_z': qc_z}

    def predict_step(self, batch, batch_idx):
        cell_embeddings = self.qz_x(torch.concat([batch['x'].log1p(), batch['s']], dim=1)).mean
        cluster_probabilities = self.qc_z(self.log_pz_c(cell_embeddings)).probs
        return cell_embeddings, cluster_probabilities
