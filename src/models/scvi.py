from typing import Any, Literal

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from torch import Tensor
from torch.distributions import Distribution, MultivariateNormal

from src.data import scRNADataModule
from src.priors import Prior
from src.priors.crypt.utils import sample_u_initial

from .abstract import DeepLatentVariableModel


class scVI(DeepLatentVariableModel):

    def __init__(
        self,
        datamodule: scRNADataModule,
        encoder_params: dict[str, Any],
        decoder_params: dict[str, Any],
        prior_params: dict[str, Any],
        encode_covariates: bool = True,
        u_initial_strategy: Literal["data", "stratified_data", "random"] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_batches = len(datamodule.batch_keys)
        self.num_genes = datamodule.num_genes

        self.encode_covariates = encode_covariates
        self.encoder: torch.nn.Module = instantiate(
            encoder_params,
            num_batches=self.num_batches,
            num_genes=self.num_genes,
            encode_covariates=encode_covariates,
            **kwargs,
        )

        self.decoder: torch.nn.Module = instantiate(
            decoder_params,
            num_batches=self.num_batches,
            num_genes=self.num_genes,
            **kwargs,
        )

        if "crypt" in prior_params["_target_"]:
            if u_initial_strategy in ["data", "stratified_data"]:
                u_sample = sample_u_initial(
                    datamodule,
                    k=prior_params["k"],
                    u_keys=["x", "s"],
                    stratification_key=(
                        "c" if u_initial_strategy == "stratified_data" else None
                    ),
                )
                u_x = u_sample["x"].log1p()
                u_s = F.one_hot(u_sample["s"], num_classes=self.num_batches).float()

            elif u_initial_strategy == "random":
                u_x = torch.randn(prior_params["k"], self.num_genes)
                u_s = F.softmax(torch.randn(prior_params["k"], self.num_batches), dim=1)

            else:
                raise ValueError(
                    f"Invalid u_initial for {prior_params['_target_']}: {u_initial_strategy}"
                )

            u_initial = (
                torch.concat([u_x, u_s], dim=-1) if self.encode_covariates else u_x
            )

        else:
            u_initial = None

        self.prior: Prior = instantiate(
            prior_params,
            encoder=self.encoder,
            num_data=datamodule.num_cells,
            u_initial=u_initial,
            **kwargs,
        )

    def qz(self, *, x: Tensor, s: Tensor, **kwargs) -> MultivariateNormal:
        x = x.log1p()
        if self.encode_covariates:
            s = F.one_hot(s, num_classes=self.num_batches).float()
            return self.encoder(torch.concat([x, s], dim=-1))
        else:
            return self.encoder(x)

    def px(self, *, z: Tensor, s: Tensor, l: Tensor, **kwargs) -> Distribution:
        s = F.one_hot(s, num_classes=self.num_batches).float()
        if z.ndim > s.ndim:
            s = s.unsqueeze(0).expand(z.shape[0], -1, -1)
        zs = torch.concat([z, s], dim=-1)
        return self.decoder(zs, l)

    def loss(self, batch: dict[str, Tensor], *, epoch: int, global_step: int) -> Tensor:
        x, s, l = batch["x"], batch["s"], batch["l"]
        qz_xs = self.qz(x=x, s=s)
        z_sample = qz_xs.rsample()
        px_zsl = self.px(z=z_sample, s=s, l=l)
        beta = self.beta(epoch, global_step)
        elbo = px_zsl.log_prob(x).mean() - beta * self.prior.dkl(qz_xs, z_sample)
        return -elbo
