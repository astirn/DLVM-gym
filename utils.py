import lightning
import lightning.pytorch.callbacks as callbacks

def get_trainer(checkpoint_metrics: list[str], checkpoint_dir: str) -> lightning.Trainer:

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
                min_delta=0.01,
                monitor='train_elbo',
                mode='max',
                patience=500,
                verbose=True
            ),
        ] + metric_checkpoint_callbacks,
        devices=1,
        gradient_clip_algorithm='value',
        gradient_clip_val=1.0,
        logger=False,
        max_epochs=-1,
        num_sanity_val_steps=0,
    )
