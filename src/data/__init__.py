from .gaussian_mixture import GaussianMixtureDataModule
from .mnist import MNISTDataModule
from .scRNA import scRNADataModule

__all__ = [
    "GaussianMixtureDataModule",
    "MNISTDataModule",
    "scRNADataModule",
]
