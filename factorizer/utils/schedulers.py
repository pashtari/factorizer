import math

from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_epochs` training epochs.
        Decreases learning rate from 1. to 0. over remaining epochs.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):

        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            factor = epoch / self.warmup_epochs
        else:
            progress = epoch - self.warmup_epochs  # progress after warmup
            w = math.pi / (self.total_epochs - self.warmup_epochs)  # frequency
            factor = 0.5 * (1.0 + math.cos(w * progress))

        return factor

