import torch


def multivariate_digamma(a: torch.Tensor, p: int) -> torch.Tensor:
    return torch.stack([torch.digamma(a + (1 - i) / 2) for i in range(1, p + 1)]).sum(
        dim=0
    )


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))
