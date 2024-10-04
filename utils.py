import numpy
import lightning
import lightning.pytorch.callbacks as callbacks
import random
import torch

from torch.utils.data import DataLoader, Dataset


def get_data_loaders(data: Dataset | list[Dataset], batch_size: int, collate: callable, seed: int = None):
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    dl_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate,
        'generator': g,
        'num_workers': 4,
        'persistent_workers': True,
        'pin_memory': True,
        'worker_init_fn': seed_worker if seed is not None else None,
    }

    return tuple(DataLoader(d, **dl_kwargs) for d in data) if isinstance(data, list) else DataLoader(data, **dl_kwargs)


def get_trainer(
        early_stopping_metric: str = None,
        checkpoint_metrics: list[str] = None,
        checkpoint_dir: str = None,
        min_delta: float = 1.0,
        patience: int = None,
) -> lightning.Trainer:

    trainer_callbacks = [
        callbacks.ModelCheckpoint(
            auto_insert_metric_name=False,
            dirpath=checkpoint_dir,
            enable_version_counter=False,
            filename=f'best_{metric}',
            mode='max',
            monitor=metric,
            save_last=True,
            save_top_k=1,
        )
        for metric in checkpoint_metrics
    ] if checkpoint_metrics is not None else []
    if early_stopping_metric is not None:
        trainer_callbacks += [
            callbacks.EarlyStopping(
                min_delta=min_delta,
                monitor=early_stopping_metric,
                mode='max',
                patience=patience,
                verbose=True
            )
        ]

    return lightning.Trainer(
        callbacks=trainer_callbacks,
        devices=1,
        enable_checkpointing=checkpoint_dir is not None,
        logger=False,
        max_epochs=-1,
        num_sanity_val_steps=0,
    )


def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
