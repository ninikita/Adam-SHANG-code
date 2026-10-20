import torch
from torch.optim.optimizer import Optimizer

class SHANG_PlusPlus(Optimizer):
    r"""
            SHANG++ baseline optimizer.

            This optimizer is included as an external baseline in our experiments.
            The method follows SHANG++ from:

                Yaxin Yu, Long Chen, and Minfu Feng.
                "SHANG++: Robust Stochastic Acceleration under Multiplicative Noise."
                arXiv:2603.09355, 2026.

            We use this implementation only for empirical comparison with Adam-SHANG
            and Adam-SHANG-s. It is not a proposed method of this paper.

            Per-step update:
                alpha = lr
                gamma = time_scale
                rho = rho
                modified_alpha = alpha / (1 + rho * alpha)
                beta = alpha / gamma

                v_{k+1} = v_k - beta * g_k

                x_{k+1} = 1 / (1 + modified_alpha) * x_k
                          + modified_alpha / (1 + modified_alpha) * v_{k+1}
                          - beta * modified_alpha / (1 + modified_alpha) * g_k

            Args:
                params:
                    Iterable of parameters to optimize.
                lr:
                    Base stepsize alpha.
                time_scale:
                    Time-scaling parameter gamma used in beta = alpha / gamma.
                rho:
                    Damping parameter used in modified_alpha = alpha / (1 + rho * alpha).
                weight_decay:
                    L2 weight decay coefficient. If nonzero, weight decay is added to
                    the gradient before the SHANG++ update.
            """
    def __init__(self, params, lr=0.5, time_scale=3,rho=1, weight_decay=0):
        defaults = dict(lr=lr, time_scale=time_scale, rho =rho, weight_decay=weight_decay)
        super(SHANG_PlusPlus, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            alpha = group['lr']
            gamma = group['time_scale']
            rho = group['rho']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if wd != 0:
                    grad.add_(p.data, alpha=wd)
                state = self.state.setdefault(p, {})
                if 'vel_prev' not in state:
                    state['vel_prev'] = p.data.clone()
                    continue
                vel_prev = state['vel_prev']
                mod_alpha = alpha/(1+ rho * alpha)
                beta = alpha / gamma
                vel_prev.add_(grad, alpha=-beta)
                p.data.mul_(1/(1+mod_alpha)).add_(vel_prev, alpha=(mod_alpha/(1+mod_alpha))).add_(grad, alpha=-beta*(mod_alpha/(1+mod_alpha)))
        return loss

