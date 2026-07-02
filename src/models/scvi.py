from typing import Any

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
        stratified_u_initial: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_batches = len(datamodule.batch_keys)
        self.num_genes = datamodule.num_genes
        self.encoder: torch.nn.Module = instantiate(
            encoder_params,
            num_batches=self.num_batches,
            num_genes=self.num_genes,
            **kwargs,
        )
        self.decoder: torch.nn.Module = instantiate(
            decoder_params,
            num_batches=self.num_batches,
            num_genes=self.num_genes,
            **kwargs,
        )

        if "crypt" in prior_params["_target_"]:
            u_initial = sample_u_initial(
                datamodule,
                k=prior_params["k"],
                u_keys=["x", "s"],
                stratification_key="c" if stratified_u_initial else None,
            )
            u_initial = torch.concat(
                [
                    u_initial["x"].log1p(),
                    F.one_hot(u_initial["s"], num_classes=self.num_batches).float(),
                ],
                dim=1,
            )
        else:
            u_initial = None

        self.prior: Prior = instantiate(
            prior_params,
            dim_x=self.num_genes + self.num_batches,
            encoder=self.encoder,
            num_data=datamodule.num_cells,
            u_initial=u_initial,
            **kwargs,
        )

    def qz(self, *, x: Tensor, s: Tensor, **kwargs) -> MultivariateNormal:
        s = F.one_hot(s, num_classes=self.num_batches).float()
        xs = torch.concat([x.log1p(), s], dim=-1)
        return self.encoder(xs)

    def px(self, *, z: Tensor, s: Tensor, l: Tensor, **kwargs) -> Distribution:
        s = F.one_hot(s, num_classes=self.num_batches).float()
        if z.ndim > s.ndim:
            s = s.unsqueeze(0).expand(z.shape[0], -1, -1)
        zs = torch.concat([z, s], dim=-1)
        return self.decoder(zs, l)

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        x, s, l = batch["x"], batch["s"], batch["l"]
        qz_xs = self.qz(x=x, s=s)
        z_sample = qz_xs.rsample()
        px_zsl = self.px(z=z_sample, s=s, l=l)
        elbo = px_zsl.log_prob(x).mean() - self.prior.dkl(qz_xs, z_sample)
        return -elbo
