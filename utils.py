import numpy
import lightning
import lightning.pytorch.callbacks as callbacks
import random
import torch


def get_trainer(
        checkpoint_metrics: list[str],
        checkpoint_dir: str | None,
        patience: int,
) -> lightning.Trainer:

    metric_checkpoint_callbacks = [
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
    ]

    return lightning.Trainer(
        callbacks=[
            callbacks.EarlyStopping(
                min_delta=1.0,
                monitor='val_elbo',
                mode='max',
                patience=patience,
                verbose=True
            ),
        ] + metric_checkpoint_callbacks,
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
