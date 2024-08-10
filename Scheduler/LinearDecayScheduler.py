from torch.optim.lr_scheduler import _LRScheduler
class LinearDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, start_epoch, end_epoch, start_lr, end_lr):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_lr = start_lr
        self.end_lr = end_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.start_epoch:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self.last_epoch > self.end_epoch:
            return [self.end_lr for _ in self.base_lrs]
        else:
            delta_epoch = self.last_epoch - self.start_epoch
            total_epochs = self.end_epoch - self.start_epoch
            decay_factor = (self.start_lr - self.end_lr) / total_epochs
            return [max(base_lr - delta_epoch * decay_factor, self.end_lr) for base_lr in self.base_lrs]
