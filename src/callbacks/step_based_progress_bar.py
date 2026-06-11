from lightning.pytorch.callbacks import TQDMProgressBar


class StepBasedProgressBar(TQDMProgressBar):
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        bar = self.train_progress_bar
        bar.reset(total=trainer.max_steps)
        bar.initial = trainer.global_step
        bar.n = trainer.global_step

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        bar = self.train_progress_bar
        if self._should_update(trainer.global_step, bar.total):
            bar.n = trainer.global_step
            bar.set_postfix(self.get_metrics(trainer, pl_module))
