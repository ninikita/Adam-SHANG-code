"""
Adam-SHANG optimizers.

This file implements two Adam-type optimizers used in the experiments.
Both optimizers maintain, for each parameter tensor, two state variables:
    y : auxiliary momentum-like variable,
    P : diagonal adaptive preconditioner with the same shape as the parameter.

The scalar stepsize alpha_k is computed once per parameter group using the
trace-ratio rule

    alpha_k = lr * sqrt( sum_i P_i^{-1} / sum_i P_i^{-2} ),
    Here, lr is the parameter lambda in Algorithm 1.

where the sums are taken over all active parameters in the group. The value is
clipped by alpha_clip for numerical stability.

The implementation uses decoupled weight decay:
    p <- (1 - lr * weight_decay) p

before the Adam-SHANG update. Sparse gradients are not supported.

This code is intended for reproducing the optimization experiments in the paper.
Default hyperparameters are provided for convenience; the exact values used in
each experiment are specified in the corresponding experiment scripts or README.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Iterable, Optional, Callable
import math


class AdamSHANG(Optimizer):
    r"""
    Adam-SHANG optimizer.

    Per-step update:
        y_k = y_{k-1} - alpha_{k-1} * P_{k-1}^{-1} * g_k

        B_k = beta * P_{k-1}^{-1} * g_k

        P_k = 1 / (1 + alpha_{k-1}) * P_{k-1}
              + alpha_{k-1} / (1 + alpha_{k-1})
                * gamma * P_{k-1}^{-1} * g_k^2

        alpha_k = lr * sqrt(Tr(P_k^{-1}) / Tr(P_k^{-2}))
        Here, lr is the parameter lambda in Algorithm 1.

        x_{k+1} = 1 / (1 + alpha_k) * x_k
                  + alpha_k / (1 + alpha_k) * y_k
                  - alpha_k / (1 + alpha_k) * B_k

    Args:
        params:
            Iterable of parameters to optimize.
        lr:
            Base scaling factor for the trace-ratio stepsize alpha_k.
        beta:
            Strength of the correction term B_k.
        gamma:
            Strength of the diagonal preconditioner injection term.
        weight_decay:
            Decoupled weight decay coefficient.
        init_P:
            Initial value of the diagonal preconditioner P.
        eps:
            Small positive constant used to avoid division by zero.
        alpha_clip:
            Upper bound on alpha_k for numerical stability.

    """

    def __init__(self, params, lr=1e-1, beta=0.05, gamma=0.05,
                 weight_decay=1e-2, init_P=1, eps=1e-6, alpha_clip: float = 0.99):
        init_alpha_prev = lr * math.sqrt(init_P)
        defaults = dict(lr=lr, beta=beta, gamma=gamma, weight_decay=weight_decay,
                        init_P=init_P, eps=eps, init_alpha_prev=init_alpha_prev, alpha_clip=alpha_clip)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr, beta, gamma = float(group["lr"]), float(group["beta"]), float(group["gamma"])
            wd, eps = float(group["weight_decay"]), float(group["eps"])
            init_P = float(group["init_P"])
            init_alpha_prev = float(group["init_alpha_prev"])
            alpha_clip = float(group["alpha_clip"])

            if "alpha_prev" not in group:
                group["alpha_prev"] = init_alpha_prev
            alpha_prev = float(group["alpha_prev"])

            global_inv1 = global_inv2 = None
            active_params, active_B_diag_gs = [], []

            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Adam-SHANG does not support sparse gradients.")
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state["y"] = p.detach().clone(memory_format=torch.preserve_format)
                    state["P"] = torch.full_like(p, init_P, memory_format=torch.preserve_format)

                y, P = state["y"], state["P"]
                P_old = P.clone(memory_format=torch.preserve_format)
                P_old_s = P_old + eps

                invP_g = g / P_old_s
                y.add_(invP_g, alpha=-alpha_prev)

                shrink = 1.0 / (1.0 + alpha_prev)
                inj_gain = gamma * alpha_prev / (1.0 + alpha_prev)
                P.mul_(shrink).add_(g.square() / P_old_s, alpha=inj_gain)

                B_diag_g = beta * invP_g

                if not torch.isfinite(y).all(): raise RuntimeError("Adam-SHANG: y inf/nan")
                if not torch.isfinite(P).all(): raise RuntimeError("Adam-SHANG: P inf/nan")

                invP = (P + eps).reciprocal()
                cur1, cur2 = invP.sum(), invP.square().sum()
                global_inv1 = cur1 if global_inv1 is None else global_inv1 + cur1
                global_inv2 = cur2 if global_inv2 is None else global_inv2 + cur2

                active_params.append(p)
                active_B_diag_gs.append(B_diag_g)

            if global_inv1 is None: continue

            ratio = (global_inv1.clamp_min(eps) / global_inv2.clamp_min(eps)).clamp_min(eps)
            alpha_k = float((lr * torch.sqrt(ratio)).item())
            alpha_k = min(alpha_k, alpha_clip)
            denom, coeff = 1.0 + alpha_k, alpha_k / (1.0 + alpha_k)

            for p, B in zip(active_params, active_B_diag_gs):
                y = self.state[p]["y"]
                p.div_(denom).add_(y, alpha=coeff).add_(B, alpha=-coeff)
                if not torch.isfinite(p).all():
                    raise RuntimeError("Adam-SHANG: param inf/nan after update")

            group["alpha_prev"] = alpha_k
        return loss


class AdamSHANGs(Optimizer):
    r"""
    Adam-SHANG-s optimizer.

    Per-step update:
        t_alpha_{k-1} = alpha_{k-1} / (1 + alpha_{k-1})

        P_k = ((1 - t_alpha_{k-1}) / 2) * P_{k-1}
              + 0.5 * sqrt(
                    (1 - t_alpha_{k-1})^2 * P_{k-1}^2
                    + 4 * t_alpha_{k-1} * gamma * g_k^2
                )

        y_k = y_{k-1} - t_alpha_{k-1} * P_k^{-1} * g_k

        alpha_k = lr * sqrt(Tr(P_k^{-1}) / Tr(P_k^{-2}))
        Here, lr is the parameter lambda in Algorithm 2.

        B_k = beta * P_k^{-1} * g_k

        x_{k+1} = 1 / (1 + alpha_k) * x_k
                  + alpha_k / (1 + alpha_k) * y_k
                  - alpha_k / (1 + alpha_k) * B_k

    Args:
        params:
            Iterable of parameters to optimize.
        lr:
            Base scaling factor for the trace-ratio stepsize alpha_k.
        beta:
            Strength of the correction term B_k.
        gamma:
            Strength of the diagonal preconditioner update.
        weight_decay:
            Decoupled weight decay coefficient.
        init_P:
            Initial value of the diagonal preconditioner P.
        eps:
            Small positive constant used to avoid division by zero.
        alpha_clip:
            Upper bound on alpha_k for numerical stability.

    """

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 1e-1,
            beta: float = 0.1,
            gamma: float = 1,
            weight_decay: float = 1e-2,
            init_P: float = 1,
            eps: float = 1e-10,
            alpha_clip: float = 0.99
    ):
        init_alpha_prev = lr * math.sqrt(init_P)
        defaults = dict(
            lr=lr,
            beta=beta,
            gamma=gamma,
            weight_decay=weight_decay,
            init_P=init_P,
            init_alpha_prev=init_alpha_prev,
            eps=eps,
            alpha_clip=alpha_clip
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta = float(group["beta"])
            gamma = float(group["gamma"])
            wd = float(group["weight_decay"])
            init_P = float(group["init_P"])
            init_alpha_prev = float(group["init_alpha_prev"])
            eps = float(group["eps"])
            alpha_clip = float(group["alpha_clip"])

            if "alpha_prev" not in group:
                group["alpha_prev"] = init_alpha_prev
            alpha_prev = float(group["alpha_prev"])

            global_inv1 = None
            global_inv2 = None

            active_params = []
            active_B_diag_gs = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Adam-SHANG-s does not support sparse gradients.")

                # decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state["y"] = p.detach().clone(memory_format=torch.preserve_format)
                    state["P"] = torch.full_like(
                        p, init_P, memory_format=torch.preserve_format
                    )

                y = state["y"]
                P = state["P"]

                P_old = P.clone(memory_format=torch.preserve_format)

                talpha_prev = alpha_prev / (1.0 + alpha_prev)
                one_minus_talpha_prev = max(1.0 - talpha_prev, eps)

                radicand = (
                        (one_minus_talpha_prev ** 2) * P_old.square()
                        + 4.0 * talpha_prev * gamma * g.square()
                )

                P_new = 0.5 * one_minus_talpha_prev * P_old + 0.5 * torch.sqrt(radicand)
                P_new.clamp_(min=eps ** 0.5)
                P.copy_(P_new)

                P_safe = P + eps
                invP = P_safe.reciprocal()

                invP_g = g * invP
                y.add_(invP_g, alpha=-talpha_prev)

                B_diag_g = beta * invP_g

                if not torch.isfinite(P).all():
                    raise RuntimeError("Adam-SHANG-s: P contains inf/nan.")
                if not torch.isfinite(y).all():
                    raise RuntimeError("Adam-SHANG-s: y contains inf/nan.")

                cur_inv1 = invP.sum()
                cur_inv2 = invP.square().sum()

                if global_inv1 is None:
                    global_inv1 = cur_inv1
                    global_inv2 = cur_inv2
                else:
                    global_inv1 = global_inv1 + cur_inv1
                    global_inv2 = global_inv2 + cur_inv2

                active_params.append(p)
                active_B_diag_gs.append(B_diag_g)

            if global_inv1 is None:
                continue

            global_inv1 = global_inv1.clamp_min(eps)
            global_inv2 = global_inv2.clamp_min(eps)
            ratio = (global_inv1 / global_inv2).clamp_min(eps)

            alpha_k_group = float((lr * torch.sqrt(ratio)).item())
            alpha_k_group = min(alpha_k_group, alpha_clip)

            denom = 1.0 + alpha_k_group
            coeff = alpha_k_group / denom

            for p, B_diag_g in zip(active_params, active_B_diag_gs):
                state = self.state[p]
                y = state["y"]

                p.div_(denom)
                p.add_(y, alpha=coeff)
                p.add_(B_diag_g, alpha=-coeff)

                if not torch.isfinite(p).all():
                    raise RuntimeError("Adam-SHANG-s: parameter contains inf/nan after update.")

            group["alpha_prev"] = alpha_k_group

        return loss


class AdamSHANGProjected(Optimizer):
    r"""Adam-SHANG with projection of ``y`` onto ``||y||_inf <= R/2``."""

    def __init__(self, params, lr=1e-1, beta=0.05, gamma=0.05,
                 weight_decay=1e-2, init_P=1, eps=1e-6,
                 alpha_clip: float = 0.99, R: float = 10.0):
        init_alpha_prev = lr * math.sqrt(init_P)
        defaults = dict(lr=lr, beta=beta, gamma=gamma, weight_decay=weight_decay,
                        init_P=init_P, eps=eps, init_alpha_prev=init_alpha_prev,
                        alpha_clip=alpha_clip, R=R)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        for group in self.param_groups:
            lr, beta, gamma = float(group["lr"]), float(group["beta"]), float(group["gamma"])
            wd, eps = float(group["weight_decay"]), float(group["eps"])
            init_P = float(group["init_P"])
            init_alpha_prev = float(group["init_alpha_prev"])
            alpha_clip = float(group["alpha_clip"])
            R = float(group["R"])

            if "alpha_prev" not in group:
                group["alpha_prev"] = init_alpha_prev
            alpha_prev = float(group["alpha_prev"])

            global_inv1 = global_inv2 = None
            active_params, active_B_diag_gs = [], []

            for p in group["params"]:
                if p.grad is None: continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Adam-SHANG does not support sparse gradients.")
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state["y"] = p.detach().clone(memory_format=torch.preserve_format)
                    state["P"] = torch.full_like(p, init_P, memory_format=torch.preserve_format)

                y, P = state["y"], state["P"]
                P_old = P.clone(memory_format=torch.preserve_format)
                P_old_s = P_old + eps

                invP_g = g / P_old_s
                y.add_(invP_g, alpha=-alpha_prev)
                y.clamp_(min=-R / 2.0, max=R / 2.0)

                shrink = 1.0 / (1.0 + alpha_prev)
                inj_gain = gamma * alpha_prev / (1.0 + alpha_prev)
                P.mul_(shrink).add_(g.square() / P_old_s, alpha=inj_gain)

                B_diag_g = beta * invP_g

                if not torch.isfinite(y).all(): raise RuntimeError("Adam-SHANG: y inf/nan")
                if not torch.isfinite(P).all(): raise RuntimeError("Adam-SHANG: P inf/nan")

                invP = (P + eps).reciprocal()
                cur1, cur2 = invP.sum(), invP.square().sum()
                global_inv1 = cur1 if global_inv1 is None else global_inv1 + cur1
                global_inv2 = cur2 if global_inv2 is None else global_inv2 + cur2

                active_params.append(p)
                active_B_diag_gs.append(B_diag_g)

            if global_inv1 is None: continue

            ratio = (global_inv1.clamp_min(eps) / global_inv2.clamp_min(eps)).clamp_min(eps)
            alpha_k = float((lr * torch.sqrt(ratio)).item())
            alpha_k = min(alpha_k, alpha_clip)
            denom, coeff = 1.0 + alpha_k, alpha_k / (1.0 + alpha_k)

            for p, B in zip(active_params, active_B_diag_gs):
                y = self.state[p]["y"]
                p.div_(denom).add_(y, alpha=coeff).add_(B, alpha=-coeff)
                if not torch.isfinite(p).all():
                    raise RuntimeError("Adam-SHANG: param inf/nan after update")

            group["alpha_prev"] = alpha_k
        return loss


class AdamSHANGsProjected(Optimizer):
    r"""Adam-SHANG-s with projection of ``y`` onto ``||y||_inf <= R/2``."""

    def __init__(
            self,
            params: Iterable[torch.nn.Parameter],
            lr: float = 1e-1,
            beta: float = 0.1,
            gamma: float = 1,
            weight_decay: float = 1e-2,
            init_P: float = 1,
            eps: float = 1e-10,
            alpha_clip: float = 0.99,
            R: float = 10.0
    ):
        init_alpha_prev = lr * math.sqrt(init_P)
        defaults = dict(
            lr=lr,
            beta=beta,
            gamma=gamma,
            weight_decay=weight_decay,
            init_P=init_P,
            init_alpha_prev=init_alpha_prev,
            eps=eps,
            alpha_clip=alpha_clip,
            R=R
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta = float(group["beta"])
            gamma = float(group["gamma"])
            wd = float(group["weight_decay"])
            init_P = float(group["init_P"])
            init_alpha_prev = float(group["init_alpha_prev"])
            eps = float(group["eps"])
            alpha_clip = float(group["alpha_clip"])
            R = float(group["R"])

            if "alpha_prev" not in group:
                group["alpha_prev"] = init_alpha_prev
            alpha_prev = float(group["alpha_prev"])

            global_inv1 = None
            global_inv2 = None

            active_params = []
            active_B_diag_gs = []

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Adam-SHANG-s does not support sparse gradients.")

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state["y"] = p.detach().clone(memory_format=torch.preserve_format)
                    state["P"] = torch.full_like(
                        p, init_P, memory_format=torch.preserve_format
                    )

                y = state["y"]
                P = state["P"]

                P_old = P.clone(memory_format=torch.preserve_format)

                talpha_prev = alpha_prev / (1.0 + alpha_prev)
                one_minus_talpha_prev = max(1.0 - talpha_prev, eps)

                radicand = (
                        (one_minus_talpha_prev ** 2) * P_old.square()
                        + 4.0 * talpha_prev * gamma * g.square()
                )

                P_new = 0.5 * one_minus_talpha_prev * P_old + 0.5 * torch.sqrt(radicand)
                P_new.clamp_(min=eps ** 0.5)
                P.copy_(P_new)

                P_safe = P + eps
                invP = P_safe.reciprocal()

                invP_g = g * invP
                y.add_(invP_g, alpha=-talpha_prev)
                y.clamp_(min=-R / 2.0, max=R / 2.0)

                B_diag_g = beta * invP_g

                if not torch.isfinite(P).all():
                    raise RuntimeError("Adam-SHANG-s: P contains inf/nan.")
                if not torch.isfinite(y).all():
                    raise RuntimeError("Adam-SHANG-s: y contains inf/nan.")

                cur_inv1 = invP.sum()
                cur_inv2 = invP.square().sum()

                if global_inv1 is None:
                    global_inv1 = cur_inv1
                    global_inv2 = cur_inv2
                else:
                    global_inv1 = global_inv1 + cur_inv1
                    global_inv2 = global_inv2 + cur_inv2

                active_params.append(p)
                active_B_diag_gs.append(B_diag_g)

            if global_inv1 is None:
                continue

            global_inv1 = global_inv1.clamp_min(eps)
            global_inv2 = global_inv2.clamp_min(eps)
            ratio = (global_inv1 / global_inv2).clamp_min(eps)

            alpha_k_group = float((lr * torch.sqrt(ratio)).item())
            alpha_k_group = min(alpha_k_group, alpha_clip)

            denom = 1.0 + alpha_k_group
            coeff = alpha_k_group / denom

            for p, B_diag_g in zip(active_params, active_B_diag_gs):
                state = self.state[p]
                y = state["y"]

                p.div_(denom)
                p.add_(y, alpha=coeff)
                p.add_(B_diag_g, alpha=-coeff)

                if not torch.isfinite(p).all():
                    raise RuntimeError("Adam-SHANG-s: parameter contains inf/nan after update.")

            group["alpha_prev"] = alpha_k_group

        return loss
