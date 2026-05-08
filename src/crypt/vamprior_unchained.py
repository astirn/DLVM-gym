import torch
import torch.distributions as dist
from torch import Tensor

from .vamprior import VampPrior


class VampPriorUnchained(VampPrior):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        qz_x = self.qz(self.u)
        if isinstance(qz_x, dist.MultivariateNormal):
            L = torch.linalg.cholesky(qz_x.covariance_matrix)
        elif isinstance(qz_x, dist.Normal):
            L = qz_x.scale.diag_embed()
        else:
            raise ValueError(f"Unsupported distribution type: {type(qz_x)}")
        self.log_L = torch.nn.Parameter(
            L.tril(-1) + torch.log(L.diagonal(dim1=-2, dim2=-1)).diag_embed()
        )

    def log_pz_c(self, z: Tensor) -> Tensor:
        mu = self.qz(self.u).loc
        L = (
            self.log_L.tril(-1)
            + (self.log_L.diagonal(dim1=-2, dim2=-1)).exp().diag_embed()
        )
        pz_c = dist.MultivariateNormal(loc=mu, scale_tril=L)
        return pz_c.log_prob(z[:, None])
