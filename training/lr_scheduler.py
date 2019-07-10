import torch.optim as optim

class CustomReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epochs_passed = 1
            epoch = self.last_epoch = self.last_epoch + 1
        else:
            epochs_passed = epoch - max(self.last_epoch, 0)
        
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += epochs_passed

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0