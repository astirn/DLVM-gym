import torch
import torch.nn.functional as F


def multivariate_digamma(a: torch.Tensor, p: int) -> torch.Tensor:
    return torch.stack([torch.digamma(a + (1 - i) / 2) for i in range(1, p + 1)]).sum(dim=0)


def scale_tril(x: torch.Tensor) -> torch.Tensor:
    return x.tril(-1) + (F.softplus(x.diagonal(dim1=-2, dim2=-1)) + 1e-5).diag_embed()


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))
