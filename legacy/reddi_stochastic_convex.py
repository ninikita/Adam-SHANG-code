#!/usr/bin/env python3
"""Two-point stochastic convex benchmark adapted from the supplied script.

    min_{-1 <= x <= 1} F(x) = E[g*x] = 0.2*x,
    g = 1010 with probability 0.01, otherwise -10.

The two SHANG updates and their coupling formulas come from the supplied
benchmark. This is a PROJECTED adaptation, not a new convergence theorem.
The default five curves are Adam-SHANG, Adam-SHANG-s, Adam, AMSGrad, and
Adam-SHANG-s with beta=0 and gamma=1. The last variant disables the gradient
correction (eta=0) and fixes gamma=1, retaining the same stepsize rule.

Run:
    python reddi_stochastic_convex.py
    python reddi_stochastic_convex.py --rules trace spectral
    python reddi_stochastic_convex.py --save-trajectories

Dependencies: numpy, matplotlib. All arithmetic/state arrays use float64.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Exact problem constants. In particular, do not replace the oracle by MU.
PROB_POS = 0.01
G_POS, G_NEG = 1010.0, -10.0
MU = 0.2
SECOND_MOMENT = 10300.0
VARIANCE = 10299.96
SIGMA1_SQ = 257499.0
NOISE_FACTOR = 1.0 + SIGMA1_SQ  # E[g^2] / |F'(x)|^2 = 257500
LOWER, UPPER, X_STAR = -1.0, 1.0, -1.0

# Defaults inherited from the supplied script except for the new problem,
# NR=30, beta2=0.99, and the explicit constrained-problem modifications.
@dataclass(frozen=True)
class Config:
    t_main: int = 100_000       # Number of updates; x_0 is also stored.
    t_gs: int = 30_000         # Independent pilot experiment for LR selection.
    nr: int = 30
    nr_tune: int = 30
    lr_grid: tuple[float, ...] = (1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0)
    beta1: float = 0.9
    beta2: float = 0.99
    warmup: int = 100
    eps: float = 1e-8
    L: float = 1.0             # Valid POSITIVE Lipschitz upper bound, not Hessian.
    R: float = 2.0             # H_R=[-R/2,R/2]=[-1,1]; |y-x*| <= R.
    x0: float = 1.0
    y0: float = 0.0
    seed_main: int = 42
    seed_tune: int = 1042
    rules: tuple[str, ...] = ("trace",)
    trace_aggregation: str = "independent"
    save_trajectories: bool = False

    def validate(self) -> None:
        for key in ("t_main", "t_gs", "nr", "nr_tune", "warmup"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.L <= 0 or not math.isfinite(self.L):
            raise ValueError("L must be a finite positive smoothness upper bound")
        if self.R != 2.0:
            raise ValueError("This benchmark fixes R=2 so H_R is exactly [-1,1]")
        if not (0 < self.eps < 1):
            raise ValueError("eps must lie in (0,1)")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1):
            raise ValueError("The momentum coefficients must lie in [0,1)")
        if not self.lr_grid or any(lr <= 0 or not math.isfinite(lr) for lr in self.lr_grid):
            raise ValueError("lr_grid must contain finite positive values")
        if not (LOWER <= self.x0 <= UPPER and LOWER <= self.y0 <= UPPER):
            raise ValueError("Initial x and y must be feasible")
        if not self.rules or any(r not in ("trace", "spectral") for r in self.rules):
            raise ValueError("rules must contain trace and/or spectral")
        if self.trace_aggregation not in ("independent", "ensemble"):
            raise ValueError("Unknown trace aggregation")
        if min(self.seed_main, self.seed_tune) < 0 or self.seed_main == self.seed_tune:
            raise ValueError("Use two distinct nonnegative seeds for tuning and evaluation")


def F(x: np.ndarray) -> np.ndarray:
    return MU * np.asarray(x)


def objective_gap(x: np.ndarray) -> np.ndarray:
    """Exact population objective gap, not a sampled loss or a regret."""
    return MU * (np.asarray(x) - X_STAR)


def sample_gradients(nsteps: int, nr: int, seed: int) -> np.ndarray:
    """Separate RNG child stream for each run; extra final sample for SHANG.

    Shape: (nsteps+1, nr). Within each run, all optimizers replay the same
    sequence. Different runs have different child streams. No sample mean
    or enforced frequency is used to generate the oracle.
    """
    children = np.random.SeedSequence(seed).spawn(nr)
    g = np.empty((nsteps + 1, nr), dtype=np.float64)
    for r, child in enumerate(children):
        rng = np.random.default_rng(child)
        g[:, r] = np.where(rng.random(nsteps + 1) < PROB_POS, G_POS, G_NEG)
    return g


def learning_rates(nsteps: int, lr0: float, warmup: int) -> np.ndarray:
    """Exactly the supplied schedule with t=k+1: lr0*min(t,W)/W/sqrt(t)."""
    t = np.arange(1, nsteps + 1, dtype=np.float64)
    return lr0 * np.minimum(t, warmup) / warmup / np.sqrt(t)


def _check_gradients(g: np.ndarray, nsteps: int) -> None:
    if g.ndim != 2 or g.shape[0] < nsteps + 1 or g.shape[1] < 1:
        raise ValueError("Expected gradients with shape (at least nsteps+1, nr)")
    if not np.isfinite(g).all():
        raise ValueError("All gradient samples must be finite")


def adam_opt(g: np.ndarray, nsteps: int, cfg: Config, lr0: float,
             *, amsgrad: bool = False) -> np.ndarray:
    """Source Adam + projection; AMSGrad adds the max of RAW second moments.

    Both moments are bias-corrected. v_0=eps and the sqrt(max(vhat,eps))
    guard are kept from the uploaded Adam implementation. For AMSGrad,
    vmax=max(vmax,v) is applied BEFORE division by 1-beta2**t.
    """
    _check_gradients(g, nsteps)
    nr = g.shape[1]
    x = np.full(nr, cfg.x0, dtype=np.float64)
    m = np.zeros(nr, dtype=np.float64)
    v = np.full(nr, cfg.eps, dtype=np.float64)
    vmax = v.copy()
    out = np.empty((nsteps + 1, nr), dtype=np.float64)
    out[0] = x
    rates = learning_rates(nsteps, lr0, cfg.warmup)
    b1pow = b2pow = 1.0
    for k in range(nsteps):
        grad = g[k]
        m = cfg.beta1 * m + (1.0 - cfg.beta1) * grad
        v = cfg.beta2 * v + (1.0 - cfg.beta2) * grad**2
        b1pow *= cfg.beta1
        b2pow *= cfg.beta2
        mh = m / (1.0 - b1pow)
        if amsgrad:
            vmax = np.maximum(vmax, v)
            vh = vmax / (1.0 - b2pow)
        else:
            vh = v / (1.0 - b2pow)
        denominator = np.sqrt(np.maximum(vh, cfg.eps)) + cfg.eps
        x = np.clip(x - rates[k] * mh / denominator, LOWER, UPPER)
        out[k + 1] = x
    return out


def coupled_step(P: np.ndarray, cfg: Config, rule: str) -> np.ndarray:
    """Return alpha (SHANG) or tilde-alpha (SHANG-s), per the supplied code.

    In one dimension Tr(P^-1)/Tr(P^-2)=P in a single run (up to eps guards).
    independent: evaluate the source's formula with nr=1 for each run.
    ensemble: reproduce the source's shared, across-run trace estimate;
              these paths are COUPLED and are NOT 30 independent repeats.
    spectral: source spectral formula, whose safety factor is 1, not 0.5.
    """
    if rule == "trace":
        inv = 1.0 / np.maximum(P, cfg.eps)
        if cfg.trace_aggregation == "ensemble":
            surrogate = max(float(inv.mean()), cfg.eps) / max(float((inv**2).mean()), cfg.eps)
            surrogate = np.full_like(P, surrogate)
        else:
            surrogate = np.maximum(inv, cfg.eps) / np.maximum(inv**2, cfg.eps)
        return 0.5 * np.sqrt(surrogate / (2.0 * cfg.L)) / NOISE_FACTOR
    if rule == "spectral":
        return np.sqrt(P / (2.0 * cfg.L)) / NOISE_FACTOR
    raise ValueError(f"Unknown rule: {rule}")


def adam_shang(g: np.ndarray, nsteps: int, cfg: Config, *, synchronous: bool,
               rule: str = "trace", beta0_gamma1: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    """Preserve lagged/synchronous update order and eta coupling from source.

    Source adaptation: x is projected before evaluating g_{k+1}; y is
    projected to H_R=[-1,1] with R=2. The source already projected y.
    No extra clipping of alpha, noise truncation, or gradient averaging.

    beta0_gamma1 adds the synchronous ablation: eta is zero from initialization
    onward and gamma is one. Its own P still determines tilde-alpha, with the
    original implicit P solve and the same preconditioned y update.
    """
    _check_gradients(g, nsteps)
    if beta0_gamma1 and not synchronous:
        raise ValueError("The beta=0, gamma=1 ablation is a synchronous variant")
    nr = g.shape[1]
    x = np.full(nr, cfg.x0, dtype=np.float64)
    y = np.full(nr, cfg.y0, dtype=np.float64)
    P = np.full(nr, math.sqrt(SECOND_MOMENT), dtype=np.float64)
    Plag = P.copy()
    a0 = coupled_step(P, cfg, rule)
    eta = np.zeros_like(a0) if beta0_gamma1 else 2.0 * a0**2 * NOISE_FACTOR
    out = np.empty((nsteps + 1, nr), dtype=np.float64)
    out[0] = x
    nxclip = nyclip = 0
    pmin = pmax = float(P[0])
    amin, amax = float(a0.min()), float(a0.max())
    checkpoints: list[dict[str, Any]] = []
    check_at = {0, 1, 9, 99, 999, 9999, nsteps - 1}

    for k in range(nsteps):
        a = coupled_step(P, cfg, rule)
        if synchronous:
            if np.any(a <= 0) or np.any(a >= 1):
                raise FloatingPointError(
                    f"SHANG-s needs 0<tilde_alpha<1; violation at update {k+1}. "
                    "Choose a different stated L upper bound; no hidden clipping was applied."
                )
            alpha = a / (1.0 - a)
            denominator_P = P
        else:
            alpha = a
            denominator_P = Plag

        x_raw = (x + alpha * y - eta * g[k] / np.maximum(denominator_P, cfg.eps)) / (1.0 + alpha)
        x_new = np.clip(x_raw, LOWER, UPPER)
        # Same indexing as the source: cached g_k in x update, fresh g_{k+1}
        # in P/y update; g_{k+1} is cached for the next x update.
        g1 = g[k + 1]
        gamma = np.ones_like(a) if beta0_gamma1 else a / cfg.R**2
        if synchronous:
            part1 = (1.0 - a) * P
            P_new = 0.5 * (part1 + np.sqrt(part1**2 + 4.0 * a * gamma * g1**2))
            P_new = np.maximum(P_new, cfg.eps)
            y_raw = y - a * g1 / np.maximum(P_new, cfg.eps)
        else:
            y_raw = y - alpha * g1 / np.maximum(P, cfg.eps)
            P_new = (P + alpha * gamma * g1**2 / np.maximum(P, cfg.eps)) / (1.0 + alpha)
            P_new = np.maximum(P_new, cfg.eps)
        y_new = np.clip(y_raw, -cfg.R / 2.0, cfg.R / 2.0)
        eta_next = np.zeros_like(a) if beta0_gamma1 else 2.0 * a**2 * NOISE_FACTOR

        if not (np.isfinite(x_new).all() and np.isfinite(y_new).all()
                and np.isfinite(P_new).all() and np.isfinite(eta_next).all()):
            raise FloatingPointError(f"Non-finite state at update {k+1}")
        nxclip += int(np.count_nonzero((x_raw < LOWER) | (x_raw > UPPER)))
        nyclip += int(np.count_nonzero((y_raw < LOWER) | (y_raw > UPPER)))
        pmin, pmax = min(pmin, float(P_new.min())), max(pmax, float(P_new.max()))
        amin, amax = min(amin, float(a.min())), max(amax, float(a.max()))
        if k in check_at:
            checkpoints.append({
                "update": k + 1,
                "x_median": float(np.median(x_new)),
                "y_median": float(np.median(y_new)),
                "P_median": float(np.median(P_new)),
                "step_parameter_median": float(np.median(a)),
                "eta_used_median": float(np.median(eta)),
                "gamma_median": float(np.median(gamma)),
            })
        Plag = P.copy()
        P = P_new
        x, y = x_new, y_new
        eta = eta_next
        out[k + 1] = x

    return out, {
        "parameter_variant": "beta=0, gamma=1" if beta0_gamma1 else "coupled",
        "step_parameter": "tilde_alpha" if synchronous else "alpha",
        "step_parameter_initial": float(a0[0]),
        "eta_initial": 0.0 if beta0_gamma1 else float(2.0 * a0[0]**2 * NOISE_FACTOR),
        "P_initial": math.sqrt(SECOND_MOMENT),
        "step_parameter_min": amin, "step_parameter_max": amax,
        "P_min": pmin, "P_max": pmax,
        "x_projection_count": nxclip, "y_projection_count": nyclip,
        "final_y": y.tolist(), "final_P": P.tolist(),
        "checkpoints": checkpoints,
        "gradient_samples_computed": nsteps + 1,
        "gradient_samples_affecting_final_x": nsteps - 1 if beta0_gamma1 else nsteps,
    }


def grid_search(g_tune: np.ndarray, cfg: Config, *, amsgrad: bool) -> tuple[float, list[dict[str, Any]]]:
    """Original pilot criterion: tail mean + tail std of MC-MEAN loss curve.

    All candidates see the same pilot samples. Evaluation samples are
    separate. One LR is chosen for the WHOLE final curve; no curve envelopes
    or per-seed best-LR selection. Only the final reporting uses medians.
    """
    name = "AMSGrad" if amsgrad else "Adam"
    rows: list[dict[str, Any]] = []
    start = int(0.8 * (cfg.t_gs + 1))
    for lr in cfg.lr_grid:
        x = adam_opt(g_tune, cfg.t_gs, cfg, lr, amsgrad=amsgrad)
        gaps = objective_gap(x)
        tail = gaps.mean(axis=1)[start:]
        score = float(tail.mean() + tail.std(ddof=0))
        rows.append({
            "optimizer": name, "lr0": lr, "pilot_steps": cfg.t_gs,
            "pilot_runs": cfg.nr_tune, "tail_mean": float(tail.mean()),
            "tail_std_over_time": float(tail.std(ddof=0)),
            "score_mean_plus_std": score,
            "pilot_final_median_gap": float(np.median(gaps[-1])),
            "selected": False,
        })
    best = min(rows, key=lambda row: row["score_mean_plus_std"])
    best["selected"] = True
    print(f"{name:8s} pilot-selected lr0={best['lr0']:g}; score={best['score_mean_plus_std']:.8g}", flush=True)
    return float(best["lr0"]), rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(out_dir: str | Path) -> None:
    """Render the saved pointwise medians in the paper's Reddi-plot style.

    The logarithmic horizontal axis starts at k=1; initialization at k=0
    remains in the saved data. No smoothing or resampling is applied.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })

    out_dir = Path(out_dir)
    metadata = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    cfg = metadata["config"]
    summary = np.load(out_dir / "curves.npz")
    names = metadata["method_order"]
    # Match the paper's legend order without changing the stored method order.
    family_order = ("Adam", "AMSGrad", "Adam-SHANG-s", "Adam-SHANG")
    names = sorted(names, key=lambda name: (
        len(family_order) if "(beta=0, gamma=1)" in name
        else family_order.index(name.split(" (")[0])))
    k = summary["k"]
    positive_steps = k > 0
    independent = cfg["trace_aggregation"] == "independent"
    run_text = f"{cfg['nr']} runs" if independent else f"{cfg['nr']} coupled paths"
    styles = {
        "Adam": ("green", (0, ()), 2.0),
        "AMSGrad": ("blue", (0, (6, 2)), 2.0),
        "Adam-SHANG-s": ("orange", (0, (2, 1)), 2.0),
        "Adam-SHANG": ("red", (0, (4, 1, 1, 1)), 2.5),
    }
    for metric, ylabel, filename in (
        ("gap", "Median objective gap", "median_objective_gap"),
        ("x", "Median output iterate", "median_iterates"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        for name in names:
            family = name.split(" (")[0]
            color, linestyle, linewidth = styles[family]
            if "(beta=0, gamma=1)" in name:
                color, linestyle, linewidth = "purple", (0, (8, 2, 1, 2)), 2.0
            # Preserve distinct curves when both stepsize rules are requested.
            if len(cfg["rules"]) > 1 and name.endswith(" (spectral)"):
                linestyle = (0, (6, 2)) if family == "Adam-SHANG-s" else (0, (8, 2, 1, 2))
            label = name.replace("(beta=0, gamma=1)", r"($\beta=0$, $\gamma=1$)")
            ax.plot(k[positive_steps], summary[f"{name}__{metric}_median"][positive_steps],
                    label=label, color=color, linestyle=linestyle, linewidth=linewidth)
        if metric == "x":
            ax.axhline(X_STAR, color="black", linewidth=1.4, linestyle="--", alpha=0.85)
            ax.text(0.02, 0.11, r"$x^\star=-1$", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=12)
            ax.set_ylim(-1.2, 1.2)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        else:
            ax.set_ylim(0.0, 0.415)
        fig.suptitle(f"Stochastic: median over {run_text}", fontsize=16, y=0.995)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
                   bbox_to_anchor=(0.5, 0.965), ncol=2, framealpha=0.9)
        ax.set_xscale("log")
        ax.set_xlim(1, max(2, cfg["t_main"]))
        ax.set_xlabel("Steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
        legend_rows = math.ceil(len(names) / 2)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88 - 0.04 * max(0, legend_rows - 2)))
        for extension in ("pdf", "png", "svg"):
            fig.savefig(out_dir / f"{filename}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    summary.close()


def run_experiment(cfg: Config, out_dir: str | Path, *, plot: bool = True) -> dict[str, Any]:
    cfg.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Problem: min_{[-1,1]} F(x)=0.2*x, x*=-1, g in {1010,-10}", flush=True)
    print(f"sigma1^2={SIGMA1_SQ:g}, E[g^2]={SECOND_MOMENT:g}, L_bound={cfg.L:g}, R={cfg.R:g}", flush=True)
    print(f"main={cfg.t_main} updates x {cfg.nr} runs; pilot={cfg.t_gs} x {cfg.nr_tune}", flush=True)
    if cfg.trace_aggregation == "ensemble":
        print("WARNING: ensemble trace couples runs; do not call these independent repeats.", flush=True)

    g_tune = sample_gradients(cfg.t_gs, cfg.nr_tune, cfg.seed_tune)
    selected: dict[str, float] = {}
    grid_rows: list[dict[str, Any]] = []
    for name, amsgrad in (("Adam", False), ("AMSGrad", True)):
        selected[name], rows = grid_search(g_tune, cfg, amsgrad=amsgrad)
        grid_rows.extend(rows)
    write_csv(out_dir / "learning_rate_grid_search.csv", grid_rows)
    del g_tune

    g_main = sample_gradients(cfg.t_main, cfg.nr, cfg.seed_main)
    methods: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    # Group both proposed methods first; baselines then use the same streams.
    for rule in cfg.rules:
        suffix = "" if len(cfg.rules) == 1 and rule == "trace" else f" ({rule})"
        for synchronous, base in ((False, "Adam-SHANG"), (True, "Adam-SHANG-s")):
            name = base + suffix
            methods[name], diagnostics[name] = adam_shang(
                g_main, cfg.t_main, cfg, synchronous=synchronous, rule=rule)
            print(f"{name:24s} final median gap={np.median(objective_gap(methods[name][-1])):.9g}", flush=True)
        name = "Adam-SHANG-s (beta=0, gamma=1)" + suffix
        methods[name], diagnostics[name] = adam_shang(
            g_main, cfg.t_main, cfg, synchronous=True, rule=rule, beta0_gamma1=True)
        print(f"{name:24s} final median gap={np.median(objective_gap(methods[name][-1])):.9g}", flush=True)
    for name, amsgrad in (("Adam", False), ("AMSGrad", True)):
        methods[name] = adam_opt(g_main, cfg.t_main, cfg, selected[name], amsgrad=amsgrad)
        print(f"{name:24s} final median gap={np.median(objective_gap(methods[name][-1])):.9g}", flush=True)

    # Quantiles are taken ACROSS RUNS separately at EVERY iteration.
    curves: dict[str, np.ndarray] = {"k": np.arange(cfg.t_main + 1)}
    final_rows: list[dict[str, Any]] = []
    per_run_rows: list[dict[str, Any]] = []
    for name, x in methods.items():
        if not np.isfinite(x).all() or np.any(x < LOWER) or np.any(x > UPPER):
            raise AssertionError(f"Invalid or infeasible trajectory: {name}")
        q25, median, q75 = np.quantile(x, [0.25, 0.5, 0.75], axis=1)
        for stat, value in (("q25", q25), ("median", median), ("q75", q75)):
            curves[f"{name}__x_{stat}"] = value
            # Exact for this positive affine gap; no temporal smoothing.
            curves[f"{name}__gap_{stat}"] = objective_gap(value)
        last_gap = objective_gap(x[-1])
        final_rows.append({
            "optimizer": name, "lr0": selected.get(name, ""),
            "updates": cfg.t_main, "runs": cfg.nr,
            "final_median_x": float(median[-1]),
            "final_median_gap": float(np.median(last_gap)),
            "final_mean_gap": float(last_gap.mean()),
            "final_gap_q25": float(np.quantile(last_gap, 0.25)),
            "final_gap_q75": float(np.quantile(last_gap, 0.75)),
        })
        for r in range(cfg.nr):
            per_run_rows.append({"optimizer": name, "run": r,
                                 "final_x": float(x[-1, r]), "final_gap": float(last_gap[r])})
    np.savez_compressed(out_dir / "curves.npz", **curves)
    write_csv(out_dir / "final_summary.csv", final_rows)
    write_csv(out_dir / "per_run_final.csv", per_run_rows)
    if cfg.save_trajectories:
        np.savez_compressed(out_dir / "trajectories.npz", gradients=g_main, **methods)

    # A compact, human-readable CSV, while curves.npz stores every iteration.
    checkpoints = np.unique(np.r_[0, np.geomspace(1, cfg.t_main, min(2000, cfg.t_main)).astype(int)])
    curve_rows = []
    for k in checkpoints:
        row: dict[str, Any] = {"update": int(k)}
        for name in methods:
            row[f"{name} median gap"] = float(curves[f"{name}__gap_median"][k])
        curve_rows.append(row)
    write_csv(out_dir / "median_gap_checkpoints.csv", curve_rows)

    metadata = {
        "config": asdict(cfg),
        "problem": {"prob_positive": PROB_POS, "positive_gradient": G_POS,
                    "negative_gradient": G_NEG, "mean_gradient": MU,
                    "second_moment": SECOND_MOMENT, "variance": VARIANCE,
                    "sigma0": 0.0, "sigma1_squared": SIGMA1_SQ,
                    "sigma1": math.sqrt(SIGMA1_SQ), "x_star": X_STAR,
                    "F_star": F(np.asarray(X_STAR)).item()},
        "method_order": list(methods),
        "selected_learning_rates": selected,
        "selection_rule": "Independent pilot; last 20%: mean+std of Monte-Carlo-mean population-gap curve",
        "reporting_rule": "Pointwise median across runs, not mean and not a median-selected single trajectory",
        "main_gradient_sha256": hashlib.sha256(g_main.tobytes()).hexdigest(),
        "main_positive_gradient_counts_per_run": (g_main[:cfg.t_main] == G_POS).sum(axis=0).tolist(),
        "main_independent_repeats": cfg.trace_aggregation == "independent",
        "adaptation_warning": "Projected adaptation of the supplied formulas; not a verification that its original convergence theorem covers this constrained problem.",
        "amsgrad_convention": "running max of uncorrected v, then second-moment bias correction; source v0=eps and epsilon guards retained",
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "elapsed_seconds_before_plotting": time.perf_counter() - started,
    }
    (out_dir / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "shang_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    if plot:
        make_figures(out_dir)
    print(f"Saved results to: {out_dir.resolve()} ({time.perf_counter()-started:.1f}s)", flush=True)
    return metadata


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path("reddi_stochastic_results"))
    parser.add_argument("--t-main", type=int, default=defaults.t_main)
    parser.add_argument("--t-gs", type=int, default=defaults.t_gs)
    parser.add_argument("--nr", type=int, default=defaults.nr)
    parser.add_argument("--nr-tune", type=int, default=defaults.nr_tune)
    parser.add_argument("--L", type=float, default=defaults.L, help="Positive smoothness upper bound; linear F has minimal constant 0")
    parser.add_argument("--lr-grid", type=float, nargs="+", default=defaults.lr_grid)
    parser.add_argument("--rules", nargs="+", choices=("trace", "spectral"), default=defaults.rules)
    parser.add_argument("--trace-aggregation", choices=("independent", "ensemble"), default=defaults.trace_aggregation)
    parser.add_argument("--seed-main", type=int, default=defaults.seed_main)
    parser.add_argument("--seed-tune", type=int, default=defaults.seed_tune)
    parser.add_argument("--save-trajectories", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        make_figures(args.out_dir)
        return
    cfg = Config(t_main=args.t_main, t_gs=args.t_gs, nr=args.nr, nr_tune=args.nr_tune,
                 L=args.L, lr_grid=tuple(args.lr_grid), rules=tuple(dict.fromkeys(args.rules)),
                 trace_aggregation=args.trace_aggregation, seed_main=args.seed_main,
                 seed_tune=args.seed_tune, save_trajectories=args.save_trajectories)
    run_experiment(cfg, args.out_dir, plot=not args.no_plots)


if __name__ == "__main__":
    main()
