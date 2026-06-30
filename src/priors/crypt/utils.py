import torch
from toolz import interleave, valmap
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.base import BaseDataModule


def sample_u_initial(
    datamodule: BaseDataModule,
    k: int,
    u_keys: str | list[str],
    stratification_key: str | None = None,
) -> dict[str, Tensor]:

    if isinstance(u_keys, str):
        u_keys = [u_keys]

    u_samples = {key: [] for key in u_keys}
    c_samples = []
    num_samples = 0

    dataset = datamodule.data["train"]
    if dataset is None:
        raise ValueError("Training dataset not found. Call datamodule.setup() first.")

    loader = DataLoader(
        dataset,
        batch_size=datamodule.batch_size["train"],
        shuffle=True,
        collate_fn=datamodule.collate_fn,
        num_workers=0,
    )
    train_iter = iter(loader)
    while num_samples < 10 * k:
        batch = next(train_iter)
        for key in u_keys:
            u_samples[key].append(batch[key])
        if stratification_key is not None:
            c_samples.append(batch[stratification_key])
        num_samples += batch[u_keys[0]].shape[0]

    u_samples = valmap(lambda x: torch.concat(list(x), dim=0), u_samples)

    if stratification_key is not None:
        c_samples = torch.hstack(c_samples)
        unique_c = c_samples.unique()
        for key in u_keys:
            u_c_samples = [
                u_samples[key][c_samples == c].unbind(dim=0) for c in unique_c
            ]
            u_samples[key] = torch.vstack(tuple(interleave(u_c_samples))).squeeze()

    return valmap(lambda x: x[:k], u_samples)
