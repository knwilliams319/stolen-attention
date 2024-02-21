# SECTION: Necessary imports
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts
#!SECTION

# SECTION: Cosine Annealing Learning Rate Scheduler with Warmup
class CosineWarmupRestartScheduler(optim.lr_scheduler.SequentialLR):
    '''
    Scheduler with a Linear Warmup Phase before Cosine Annealing with Restarts is applied.
    '''
    def __init__(
            self, 
            optimizer, 
            warmup_updates, 
            warmup_init_lr, 
            warmup_end_lr, 
            min_lr, 
            lr_period_updates, 
            t_mult
        ):
        super().__init__(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer=optimizer,
                    start_factor=(warmup_end_lr - warmup_init_lr)/warmup_updates,
                    total_iters=warmup_updates-1
                ),
                CosineAnnealingWarmRestarts(
                    optimizer=optimizer,
                    T_0=lr_period_updates,
                    T_mult=t_mult,
                    eta_min=min_lr
                )
            ],
            milestones=[warmup_updates-1]
        )
#!SECTION

# SECTION: REX Learning Rate Scheduler
class REXScheduler(optim.lr_scheduler._LRScheduler):
    '''
    Reflected Exponential Scheduler for good convergence results without hyperparamters to tune (besides initial LR)
    # LINK: https://arxiv.org/pdf/2107.04197.pdf
    '''
    def __init__(
            self, 
            optimizer,
            num_steps=-1,
        ):
        assert num_steps > 0, "Must train for positive number of steps, default=-1"
        self.T = num_steps  # the total number of training steps
        super().__init__(optimizer)

    def get_lr(self):
        t = self.last_epoch
        factor = (1 - t/self.T)/(0.5 + 0.5 * (1 - t/self.T))
        return [group['initial_lr'] * factor for group in self.optimizer.param_groups]
#!SECTION