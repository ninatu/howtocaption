import math


class CustomCosineSchedulerWithWarmup:
    def __init__(self, optimizer, T_max, lr, warmup_epochs=0, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min

        if isinstance(lr, list):
            self.base_lr = lr
        else:
            self.base_lr = [lr] * len(self.optimizer.param_groups)
        assert len(self.optimizer.param_groups) == len(self.base_lr)

        self.step(0)

    def step(self, epoch):
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lr):
            epoch = epoch % self.T_max
            if epoch < self.warmup_epochs:
                cur_lr = epoch / self.warmup_epochs * base_lr
            else:
                cur_lr = self.eta_min + (base_lr - self.eta_min) * 0.5 * (
                            1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)))

            param_group['lr'] = cur_lr


class SchedulerWithWarmup:
    def __init__(self, optimizer, lr, warmup_epochs=0, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min

        self.base_lr = lr
        self.step(0)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            cur_lr = epoch / self.warmup_epochs * self.base_lr
        else:
            cur_lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr


class SchedulerWithWarmupAndDecay:
    def __init__(self, optimizer, lr, warmup_epochs=0, min_lr=0, decay_rate=1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.decay_rate = decay_rate

        self.base_lr = lr
        self.step(0)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            cur_lr = epoch / self.warmup_epochs * self.base_lr
        else:
            cur_lr = max(self.min_lr, self.base_lr * (self.decay_rate ** (epoch - self.warmup_epochs)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr