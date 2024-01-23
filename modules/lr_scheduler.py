# SECTION: Necessary imports
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, ChainedScheduler, MultiplicativeLR
#!SECTION

# SECTION: Cosine Annealing Learning Rate Scheduler with Warmup
class CosineWarmupRestartScheduler(optim.lr_scheduler.SequentialLR):
    '''
    Scheduler with a Linear Warmup Phase before Cosine Annealing with Restarts is applied.
    '''
    def __init__(self, optimizer, warmup_updates, warmup_init_lr, warmup_end_lr, min_lr, lr_period_updates, t_mult, lr_shrink=None):
        super().__init__(optimizer,
                         schedulers=[LinearLR(optimizer=optimizer,
                                              start_factor=(warmup_end_lr - warmup_init_lr)/warmup_updates,
                                              total_iters=warmup_updates-1),
                                     CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                 T_0=lr_period_updates,
                                                                 T_mult=t_mult,
                                                                 eta_min=min_lr)],
                         milestones=[warmup_updates-1])
#!SECTION