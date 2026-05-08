from lightning.pytorch.callbacks import TQDMProgressBar


class StepBasedProgressBar(TQDMProgressBar):
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_progress_bar.n = trainer.global_step
        self.train_progress_bar.total = trainer.max_steps
        self.train_progress_bar.refresh()
