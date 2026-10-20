"""
Adam-SHANG Convex Benchmark  — σ₀=0 pure multiplicative-noise
================================================
Problem:
  F(x) = Σᵢ fₐ(xᵢ),  fₐ(t) = |t|^d (|t|≤1) else 1+d(|t|-1)
  F* = 0,  x* = 0,  L = d(d-1)

Noise (sigma0=0):
  gᵢ = (1 + σ₁·Zᵢ)·∂ᵢF(x),   Zᵢ ~ N(0,1) i.i.d.

Reproducibility notes:
  - This script reproduces the synthetic convex benchmark figures for
    Adam-SHANG and Adam-SHANG-s.
  - Both Adam-SHANG variants use R=1 and project y onto
    H_R = {y : ||y||_inf <= R/2}; since x*=0, this enforces
    ||y_k-x*||_inf <= R exactly.
  - No external dataset is required. The default dimension is d=16,
    as in the paper; --dims 8 16 also runs the legacy d=8 comparison.
  - The global NumPy seed is fixed below for reproducibility.
  - Spectral-step comparisons use isolated deterministic seeds, so adding
    them does not change the legacy random stream used by the other methods.
  - Trace steps are computed per trajectory with the dimension-dependent
    factor sqrt(2 / (sqrt(d) + 1)), the largest factor universally ensuring
    trace steps do not exceed spectral steps. This guarantees lagged
    stepsize admissibility only, not convergence of the synchronous variant.
  - NR denotes the number of Monte Carlo runs used to estimate E[F(x_k)].
    When sigma1=0, the experiment is deterministic and the script uses NR=1.
  - Adam's base learning rate is selected by a grid search over LR_GRID using
    a short pilot run of length T_GS.
  - The script writes objective figures and final-loss summaries to OUT_DIR.
  - Adam-SHANG and Adam-SHANG-s report F(x_k^+), including k=0;
    the SGD, SHANG, and Adam baselines still report F(x_k).
  - Metric-floor diagnostics cover initialization and every update for
    every trajectory and coordinate, before clipping. They do not change
    the updates or random draws. Summaries are saved as metric_floor_summary.
"""

import argparse
import csv
import os
from contextlib import contextmanager
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

PROJECTION_RADIUS = 1.0


@contextmanager
def isolated_numpy_seed(seed):
    """Run an added comparison without changing the legacy RNG stream."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

#  Objective function
def F(x, d):
    """x: [nr, d] → [nr]"""
    ax = np.abs(x)
    return np.sum(np.where(ax <= 1.0, ax**d, 1.0 + d*(ax - 1.0)), axis=-1)

def gradF(x, d):
    ax = np.abs(x)
    return np.where(ax <= 1.0,
                    np.where(ax > 0, d * ax**(d-2) * x, 0.0),
                    d * np.sign(x))

#  Noise oracle
def noisy_g(x, s1, d):
    gf = gradF(x, d)
    if s1 == 0.0:
        return gf
    Z = np.random.randn(*x.shape)
    return (1.0 + s1 * Z) * gf

def project_y(y, R=PROJECTION_RADIUS):
    """Projection onto H_R={y: ||y||_inf <= R/2}."""
    return np.clip(y, -R / 2.0, R / 2.0)

def save_final_loss_table(rows, out_dir):
    """Save rebuttal-ready final-loss summaries as CSV and Markdown."""
    columns = ["d", "sigma0", "sigma1", "optimizer", "evaluation_point", "lr0", "final_loss"]
    csv_path = os.path.join(out_dir, "final_loss_summary.csv")
    md_path  = os.path.join(out_dir, "final_loss_summary.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    header = "| " + " | ".join(columns) + " |"
    rule   = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines  = [header, rule]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    table = "\n".join(lines)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(table + "\n")

    print("\nFinal-loss summary:\n")
    print(table)
    print(f"\n  → {csv_path}")
    print(f"  → {md_path}")


# Metric-floor diagnostics are observational: no RNG draws or state changes.
# k=0 is initialization; update k records the metric used at that iterate.
def init_metric_floor_stats(stats, initial_second_moment, P0, eps):
    stats.clear()
    stats.update({
        "metric_floor": float(eps),
        "nr": P0.shape[0],
        "min_raw_p": float(np.min(np.sqrt(initial_second_moment))),
        # Initialization clips the second moment before taking its square root.
        "initial_floor_hits": int(np.count_nonzero(initial_second_moment < eps)),
        "initial_inverse_floor_hits": int(np.count_nonzero(P0 < eps)),
        "updates_checked": 0,
        "entries_checked": 0,
        "floor_hits": 0,
        "floor_steps": 0,
        "first_floor_k": "",
        "max_floor_lift": 0.0,
        "nonfinite_entries": int(np.count_nonzero(~np.isfinite(P0))),
        "_hit_runs": np.zeros(P0.shape[0], dtype=bool),
    })


def record_metric_floor(stats, P_raw, k):
    """Inspect each proposed metric exactly once, before np.maximum."""
    eps = stats["metric_floor"]
    stats["min_raw_p"] = float(np.minimum(stats["min_raw_p"], np.min(P_raw)))
    stats["updates_checked"] += 1
    stats["entries_checked"] += P_raw.size
    stats["nonfinite_entries"] += int(np.count_nonzero(~np.isfinite(P_raw)))
    clipped = P_raw < eps  # Equality does not change the metric.
    hits = int(np.count_nonzero(clipped))
    if hits:
        stats["floor_hits"] += hits
        stats["floor_steps"] += 1
        stats["_hit_runs"] |= np.any(clipped, axis=1)
        if stats["first_floor_k"] == "":
            stats["first_floor_k"] = k
        stats["max_floor_lift"] = max(
            stats["max_floor_lift"], float(np.max(eps - P_raw[clipped])))


def metric_floor_summary(stats):
    row = {key: value for key, value in stats.items() if not key.startswith("_")}
    row["runs_with_floor"] = int(np.count_nonzero(stats["_hit_runs"]))
    if stats["nonfinite_entries"]:
        row["floor_inactive"] = "invalid (nonfinite metric)"
    else:
        row["floor_inactive"] = (
            "yes" if stats["floor_hits"] == 0
            and stats["initial_floor_hits"] == 0
            and stats["initial_inverse_floor_hits"] == 0 else "no")
    for key in ("metric_floor", "min_raw_p", "max_floor_lift"):
        row[key] = f"{row[key]:.10e}"
    return row


def save_metric_floor_table(rows, out_dir):
    """Save all-run, all-iteration diagnostics without storing metric histories."""
    if not rows:
        return
    columns = list(rows[0])
    csv_path = os.path.join(out_dir, "metric_floor_summary.csv")
    md_path = os.path.join(out_dir, "metric_floor_summary.md")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines.extend("| " + " | ".join(str(row[key]) for key in columns) + " |"
                 for row in rows)
    notes = (
        "min_raw_p includes the unregularized RMS initialization and every "
        "proposed metric before the update floor. floor_hits counts coordinate "
        "entries strictly below metric_floor; floor_steps counts updates with "
        "any hit, and runs_with_floor counts distinct affected trajectories. "
        "Update indices start at k=1. Initialization guards are counted "
        "separately. Nonfinite metrics invalidate an inactivity claim.\n")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n\n" + notes)
    print("\nMetric-floor summary:\n")
    print("\n".join(lines))
    print(f"\n  → {csv_path}")
    print(f"  → {md_path}")

#  SGD
def sgd(d, s1, T, nr):
    L   = d * (d - 1)
    eta = 1.0 / (L * (1.0 + s1**2))
    x   = np.ones((nr, d))
    res = np.zeros(T)
    res[0] = np.mean(F(x, d))
    for k in range(T - 1):
        x = x - eta * noisy_g(x, s1, d)
        res[k+1] = np.mean(F(x, d))
    return res

#  SHANG
def shang(d, s1, T, nr):
    L  = d * (d - 1)
    R  = 1.0 + s1**2
    x  = np.ones((nr, d))
    v  = np.ones((nr, d))
    res = np.zeros(T)
    res[0] = np.mean(F(x, d))
    for k in range(T - 1):
        n     = k
        alpha = 2.0 / (n + 1)
        gam   = alpha**2 * L * R**2
        beta  = R * alpha / gam
        ap    = 2.0/n if n > 0 else alpha
        gp    = ap**2 * L * R**2 / (1.0+ap) if n > 0 else gam
        g     = noisy_g(x, s1, d)
        v_new = v - (ap / gp) * g
        x_new = (1.0/(1+alpha))*x + (alpha/(1+alpha))*v_new \
                - beta*(alpha/(1+alpha))*g
        res[k+1] = np.mean(F(x_new, d))
        x, v = x_new, v_new
    return res

#  Adam
def adam_opt(d, s1, T, nr, lr0, b1=0.9, b2=0.999, eps=1e-8, warmup=100):
    x  = np.ones((nr, d))
    m  = np.zeros((nr, d))
    v  = np.full((nr, d), eps)
    res = np.zeros(T)
    res[0] = np.mean(F(x, d))
    for k in range(T - 1):
        lr_k = lr0 * min(k + 1, warmup) / warmup / np.sqrt(k + 1)
        g    = noisy_g(x, s1, d)
        m    = b1*m + (1-b1)*g
        v    = b2*v + (1-b2)*g**2
        mh   = m / (1 - b1**(k+1))
        vh   = v / (1 - b2**(k+1))
        P    = np.sqrt(np.maximum(vh, eps))
        x    = x - lr_k * mh / (P + eps)
        res[k+1] = np.mean(F(x, d))
    return res

# Adam baseline tuning: select lr0 from LR_GRID by minimizing mean+std
# over the last 20% of a pilot run. This makes the baseline setting explicit
# for reproducibility and avoids using unseen information from the full run.
def adam_grid_search(d, s1, T_gs, nr, lrs):
    best_lr, best_val = lrs[0], np.inf
    for lr in lrs:
        r      = adam_opt(d, s1, T_gs, nr, lr)
        tail   = r[int(0.8*T_gs):]
        crit   = float(np.mean(tail) + np.std(tail))
        if crit < best_val:
            best_val, best_lr = crit, lr
    print(f"      Adam grid search → lr₀* = {best_lr:.3g}  "
          f"(mean+std = {best_val:.4g})")
    return best_lr

#  Adam-SHANG — fixed R=1 with projection onto H_R
def adam_shang(d, s1, T, nr, eps=1e-8, stepsize_rule="trace",
               metric_floor_stats=None):
    # Largest dimension-dependent factor ensuring trace steps <= spectral steps.
    trace_safety_factor = np.sqrt(2.0 / (np.sqrt(d) + 1.0))
    L    = d * (d - 1)
    s1sq = s1**2

    x    = np.ones((nr, d))
    y    = np.zeros((nr, d))

    R    = PROJECTION_RADIUS

    gf0  = gradF(x, d)
    initial_second_moment = (1.0 + s1sq) * gf0**2
    P    = np.sqrt(np.maximum(initial_second_moment, eps))
    if metric_floor_stats is not None:
        init_metric_floor_stats(metric_floor_stats, initial_second_moment, P, eps)
    Plag = P.copy()

    g    = noisy_g(x, s1, d)

    iP      = 1.0 / np.maximum(P, eps)
    if stepsize_rule == "trace":
        trPinv1 = np.sum(iP, axis=1, keepdims=True)
        trPinv2 = np.sum(iP ** 2, axis=1, keepdims=True)
        surrogate = trPinv1 / trPinv2
        alpha0 = trace_safety_factor * np.sqrt(surrogate / (2.0 * L)) / (1.0 + s1sq)
    elif stepsize_rule == "spectral":
        lambda_min = np.min(P, axis=1, keepdims=True)
        alpha0 = np.sqrt(lambda_min / (2.0 * L)) / (1.0 + s1sq)
    else:
        raise ValueError(f"Unknown stepsize rule: {stepsize_rule}")
    eta       = 2.0 * alpha0**2 * (1.0 + s1sq)

    res   = np.zeros(T)
    # x_0^+ uses eta_0=eta_1 and the initial metric.
    res[0] = np.mean(F(x - eta * iP * g, d))

    for k in range(T - 1):

        iP      = 1.0 / np.maximum(P, eps)
        if stepsize_rule == "trace":
            trPinv1 = np.sum(iP, axis=1, keepdims=True)
            trPinv2 = np.sum(iP ** 2, axis=1, keepdims=True)
            surrogate = trPinv1 / trPinv2
            alpha = trace_safety_factor * np.sqrt(surrogate / (2.0 * L)) / (1.0 + s1sq)
        else:
            lambda_min = np.min(P, axis=1, keepdims=True)
            alpha = np.sqrt(lambda_min / (2.0 * L)) / (1.0 + s1sq)

        iPlag = 1.0 / np.maximum(Plag, eps)
        x_new = (1.0/(1+alpha))*x + (alpha/(1+alpha))*y \
                - (eta/(1+alpha)) * iPlag * g
        g1 = noisy_g(x_new, s1, d)

        y_half = y - alpha * iP * g1
        y_new  = project_y(y_half, R)
        gamma  = alpha / R**2

        G2    = g1**2
        P_new = (1.0/(1+alpha))*P + (alpha*gamma/(1+alpha)) * iP * G2
        if metric_floor_stats is not None:
            record_metric_floor(metric_floor_stats, P_new, k + 1)
        P_new = np.maximum(P_new, eps)

        eta1 = 2.0 * alpha**2 * (1.0 + s1sq)
        # Lagged correction: x_{k+1}^+ uses P_k, not P_{k+1}.
        x_plus_new = x_new - eta1 * iP * g1
        res[k+1] = np.mean(F(x_plus_new, d))

        Plag    = P.copy()
        P       = P_new
        x, y, g = x_new, y_new, g1
        eta     = eta1

    return res

#  Adam-SHANG-s — fixed R=1 with projection onto H_R
def adam_shangs(d, s1, T, nr, eps=1e-8, stepsize_rule="trace",
                metric_floor_stats=None):
    # Largest dimension-dependent factor ensuring trace steps <= spectral steps.
    trace_safety_factor = np.sqrt(2.0 / (np.sqrt(d) + 1.0))
    L    = d * (d - 1)
    s1sq = s1**2

    x    = np.ones((nr, d))
    y    = np.zeros((nr, d))

    R    = PROJECTION_RADIUS

    gf0  = gradF(x, d)
    initial_second_moment = (1.0 + s1sq) * gf0**2
    P    = np.sqrt(np.maximum(initial_second_moment, eps))
    if metric_floor_stats is not None:
        init_metric_floor_stats(metric_floor_stats, initial_second_moment, P, eps)

    g    = noisy_g(x, s1, d)

    iP      = 1.0 / np.maximum(P, eps)
    if stepsize_rule == "trace":
        trPinv1 = np.sum(iP, axis=1, keepdims=True)
        trPinv2 = np.sum(iP ** 2, axis=1, keepdims=True)
        surrogate = trPinv1 / trPinv2
        talpha0 = trace_safety_factor * np.sqrt(surrogate / (2.0 * L)) / (1.0 + s1sq)
    elif stepsize_rule == "spectral":
        lambda_min = np.min(P, axis=1, keepdims=True)
        talpha0 = np.sqrt(lambda_min / (2.0 * L)) / (1.0 + s1sq)
    else:
        raise ValueError(f"Unknown stepsize rule: {stepsize_rule}")
    alpha0   = talpha0 / (1.0 - talpha0)
    eta       = 2.0 * talpha0**2 * (1.0 + s1sq)

    res   = np.zeros(T)
    res[0] = np.mean(F(x - eta * iP * g, d))

    for k in range(T - 1):

        iP      = 1.0 / np.maximum(P, eps)
        if stepsize_rule == "trace":
            trPinv1 = np.sum(iP, axis=1, keepdims=True)
            trPinv2 = np.sum(iP ** 2, axis=1, keepdims=True)
            surrogate = trPinv1 / trPinv2
            talpha = trace_safety_factor * np.sqrt(surrogate / (2.0 * L)) / (1.0 + s1sq)
        else:
            lambda_min = np.min(P, axis=1, keepdims=True)
            talpha = np.sqrt(lambda_min / (2.0 * L)) / (1.0 + s1sq)

        alpha = talpha / (1.0 - talpha)
        x_new = (1.0/(1+alpha))*x + (alpha/(1+alpha))*y \
                - (eta/(1+alpha)) * iP * g

        g1 = noisy_g(x_new, s1, d)

        G2     = g1**2
        gamma = talpha / (R**2)
        part1  = (1 - talpha) * P
        part2  = part1**2 + 4 * talpha * gamma * G2
        P_new  = 0.5 * part1 + 0.5 * np.sqrt(np.maximum(part2, 0.0))
        if metric_floor_stats is not None:
            record_metric_floor(metric_floor_stats, P_new, k + 1)
        P_new  = np.maximum(P_new, eps)

        iP_new = 1.0 / np.maximum(P_new, eps)
        y_half = y - talpha * iP_new * g1
        y_new  = project_y(y_half, R)

        eta1     = 2.0 * talpha**2 * (1.0 + s1sq)
        # Synchronous correction: x_{k+1}^+ uses the updated P_{k+1}.
        x_plus_new = x_new - eta1 * iP_new * g1
        res[k+1] = np.mean(F(x_plus_new, d))

        P       = P_new
        x, y, g = x_new, y_new, g1
        eta     = eta1

    return res

#  Experiment configuration and execution (no experiment runs on import).
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=100000,
                        help="Recorded iterates, including initialization (default: 100000).")
    parser.add_argument("--pilot-steps", type=int, default=30000,
                        help="Recorded iterates per Adam tuning trial (default: 30000).")
    parser.add_argument("--runs", type=int, default=200,
                        help="Monte Carlo trajectories; zero noise still uses one (default: 200).")
    parser.add_argument("--dims", type=int, nargs="+", default=[16],
                        help="Dimensions; the paper uses 16 (legacy comparison: --dims 8 16).")
    parser.add_argument("--output", default="./adam_shang_results_sigma0_0",
                        help="Output directory; existing same-name result files are overwritten.")
    args = parser.parse_args(argv)
    if min(args.steps, args.pilot_steps, args.runs) < 1:
        parser.error("--steps, --pilot-steps, and --runs must be positive.")
    if any(d < 2 for d in args.dims):
        parser.error("Every dimension must be at least 2.")
    T_MAIN, T_GS, NR = args.steps, args.pilot_steps, args.runs
    DS = args.dims
    OUT_DIR = args.output
    np.random.seed(42)

    SIGMA1S = [0, 10, 30]

    LR_GRID = [1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1]

    os.makedirs(OUT_DIR, exist_ok=True)

    C = {"SGD":          "black",
         "SHANG":        "blue",
         "Adam":         "green",
         "Adam-SHANG-s (Trace)": "orange",
         "Adam-SHANG (Trace)":   "red",
         "Adam-SHANG-s (spectral)": "darkgoldenrod",
         "Adam-SHANG (spectral)":   "firebrick"}

    xs = np.arange(1, T_MAIN + 1)

    # ══════════════════════════════════════════════════════════════════════════════
    #  Main loop
    # ══════════════════════════════════════════════════════════════════════════════

    print("=" * 65)
    print("  Adam-SHANG  σ₀=0")
    print(f"  d ∈ {DS},  σ₁ ∈ {SIGMA1S},  T={T_MAIN},  NR={NR}")
    print("=" * 65)

    ALL = {}
    FINAL_LOSS_ROWS = []
    METRIC_FLOOR_ROWS = []

    for d in DS:
        L = d * (d - 1)
        for s1 in SIGMA1S:
            nr = 1 if s1 == 0 else NR
            tag = f"d={d}, σ₁={s1}"
            print(f"\n{'─' * 55}\n  {tag}  (L={L}, nr={nr})\n{'─' * 55}")

            t0 = datetime.now()
            r_sgd = sgd(d, s1, T_MAIN, nr)
            print(f"  SGD          final={r_sgd[-1]:.4g}  ({(datetime.now()-t0).seconds}s)")

            t0 = datetime.now()
            r_shng = shang(d, s1, T_MAIN, nr)
            print(f"  SHANG        final={r_shng[-1]:.4g}  ({(datetime.now()-t0).seconds}s)")

            t0 = datetime.now()
            blr    = adam_grid_search(d, s1, T_GS, nr, LR_GRID)
            r_adam = adam_opt(d, s1, T_MAIN, nr, blr)
            print(f"  Adam         final={r_adam[-1]:.4g}  (lr₀={blr:.3g})")

            t0 = datetime.now()
            floor_ashs = {}
            r_ashs = adam_shangs(d, s1, T_MAIN, nr, metric_floor_stats=floor_ashs)
            print(f"  Adam-SHANG-s (Trace) final={r_ashs[-1]:.4g}  ({(datetime.now()-t0).seconds}s)")

            t0 = datetime.now()
            floor_ash = {}
            r_ash = adam_shang(d, s1, T_MAIN, nr, metric_floor_stats=floor_ash)
            print(f"  Adam-SHANG (Trace)   final={r_ash[-1]:.4g}  ({(datetime.now()-t0).seconds}s)")

            t0 = datetime.now()
            spectral_seed = 420000 + 10000 * d + 10 * s1
            floor_ashs_spectral = {}
            with isolated_numpy_seed(spectral_seed + 1):
                r_ashs_spectral = adam_shangs(
                    d, s1, T_MAIN, nr, stepsize_rule="spectral",
                    metric_floor_stats=floor_ashs_spectral)
            print(f"  Adam-SHANG-s (spectral) final={r_ashs_spectral[-1]:.4g}  "
                  f"({(datetime.now()-t0).seconds}s)")

            t0 = datetime.now()
            floor_ash_spectral = {}
            with isolated_numpy_seed(spectral_seed + 2):
                r_ash_spectral = adam_shang(
                    d, s1, T_MAIN, nr, stepsize_rule="spectral",
                    metric_floor_stats=floor_ash_spectral)
            print(f"  Adam-SHANG (spectral)   final={r_ash_spectral[-1]:.4g}  "
                  f"({(datetime.now()-t0).seconds}s)")

            for optimizer, stats in [
                    ("Adam-SHANG-s (Trace)", floor_ashs),
                    ("Adam-SHANG (Trace)", floor_ash),
                    ("Adam-SHANG-s (spectral)", floor_ashs_spectral),
                    ("Adam-SHANG (spectral)", floor_ash_spectral)]:
                summary = metric_floor_summary(stats)
                METRIC_FLOOR_ROWS.append({
                    "d": d, "sigma0": 0, "sigma1": s1, "optimizer": optimizer,
                    **summary,
                })
                print(f"  {optimizer}: min raw P={summary['min_raw_p']}, "
                      f"floor hits={summary['floor_hits']}, "
                      f"first k={summary['first_floor_k']}, "
                      f"floor inactive={summary['floor_inactive']}")

            ALL[(d, s1)] = dict(sgd=r_sgd, shang=r_shng,
                                adam=r_adam,
                                ashs=r_ashs,
                                ash=r_ash,
                                ashs_spectral=r_ashs_spectral,
                                ash_spectral=r_ash_spectral,
                                blr=blr)

            results = [("SGD", r_sgd, ""),
                       ("SHANG", r_shng, ""),
                       ("Adam", r_adam, f"{blr:.10g}"),
                       ("Adam-SHANG-s (Trace)", r_ashs, ""),
                       ("Adam-SHANG (Trace)", r_ash, ""),
                       ("Adam-SHANG-s (spectral)", r_ashs_spectral, ""),
                       ("Adam-SHANG (spectral)", r_ash_spectral, "")]
            for alg, r, lr0 in results:
                steps = [100, 1000, 10000, T_MAIN - 1]
                vals  = [f"{r[min(s, T_MAIN-1)]:.2e}" for s in steps]
                print(f"  {alg:<14} k={steps}: {vals}")
                FINAL_LOSS_ROWS.append({
                    "d": d,
                    "sigma0": 0,
                    "sigma1": s1,
                    "optimizer": alg,
                    "evaluation_point": "x_k^+" if alg.startswith("Adam-SHANG") else "x_k",
                    "lr0": lr0,
                    "final_loss": f"{float(r[-1]):.10e}",
                })

            F0 = max(r_sgd[0], 1e-15)
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            ax.set_title(rf"$d={d}$,  $\sigma_1={s1}$,  $\sigma_0=0$", fontweight="bold")
            x_left = 1.0 if s1 == 0 else np.sqrt(10.0)

            ax.loglog(xs, np.maximum(r_sgd,  1e-15), color=C["SGD"],          ls="-",  lw=1.8, label="SGD")
            ax.loglog(xs, np.maximum(r_shng, 1e-15), color=C["SHANG"],        ls="-",  lw=1.8, label="SHANG")
            ax.loglog(xs, np.maximum(r_adam, 1e-15), color=C["Adam"],         ls="-",  lw=1.8, label=rf"Adam  $\ell_0$={blr:.2g}")
            ax.loglog(xs, np.maximum(r_ashs, 1e-15), color=C["Adam-SHANG-s (Trace)"], ls="-",  lw=2.4, label="Adam-SHANG-s (Trace)", zorder=5)
            ax.loglog(xs, np.maximum(r_ash,  1e-15), color=C["Adam-SHANG (Trace)"],   ls="-",  lw=2.4, label="Adam-SHANG (Trace)",   zorder=5)
            ax.loglog(xs, np.maximum(r_ashs_spectral, 1e-15),
                      color=C["Adam-SHANG-s (spectral)"], ls="--", lw=2.0,
                      label="Adam-SHANG-s (spectral)", zorder=4)
            ax.loglog(xs, np.maximum(r_ash_spectral, 1e-15),
                      color=C["Adam-SHANG (spectral)"], ls="--", lw=2.0,
                      label="Adam-SHANG (spectral)", zorder=4)
            ax.loglog(xs, F0 / xs,      color="grey", ls=":",              lw=0.9, alpha=0.5, label=r"$O(1/k)$")
            ax.loglog(xs, F0 / xs ** 2, color="grey", ls=(0,(3,1,1,1)),    lw=0.9, alpha=0.5, label=r"$O(1/k^2)$")

            ax.set_xlabel("Iteration $k$")
            ax.set_ylabel(r"$\mathbb{E}[f(x_k^+)]$")
            ax.legend(loc="lower left", framealpha=0.92)
            ax.grid(True, which="both", alpha=0.22, ls=":")
            y_top = float(F0) * 2.0
            y_bot = max(min(r_sgd[-1], r_shng[-1], r_adam[-1],
                            r_ashs[-1], r_ash[-1], r_ashs_spectral[-1],
                            r_ash_spectral[-1]) / 100.0,
                        y_top * 1e-10)
            ax.set_ylim(bottom=y_bot, top=y_top)
            ax.set_xlim(left=x_left, right=T_MAIN)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"d{d}_s1{s1}.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"  → d{d}_s1{s1}.png")

    save_final_loss_table(FINAL_LOSS_ROWS, OUT_DIR)
    save_metric_floor_table(METRIC_FLOOR_ROWS, OUT_DIR)


    print(f"\nDone. All results are saved in {OUT_DIR}/")


if __name__ == "__main__":
    main()
