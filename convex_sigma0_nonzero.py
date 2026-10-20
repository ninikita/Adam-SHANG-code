"""
Adam-SHANG Convex Benchmark: fixed σ₁=10, varying σ₀
============================================================
Problem setting:
  d = 16, σ₁ = 10
  σ₀ in {0.5, 1.0, 3.0}

Theory-motivated coupling used in the convex benchmark:
  Adam-SHANG  : gamma_k = alpha_k / R^2
  Adam-SHANG-s: gamma_k = tilde_alpha_k / R^2
  Both methods use R=1 and project y onto
  H_R = {y : ||y||_inf <= R/2}. Since x*=0, this enforces
  ||y_k-x*||_inf <= R exactly.
  Trace steps are computed separately for each trajectory with the
  dimension-dependent choice lambda = sqrt(2/(sqrt(d)+1)).
  For the lagged method, this largest universal spectral-domination factor guarantees
  stepsize admissibility without a posteriori checks. This does not assert
  a convergence guarantee for the synchronous method or the metric floor.

Noise and Assumption 2.1(G2):
  The common oracle is
    g_i = (1 + sigma1 Z_i) grad_i F(x) + (sigma0/sqrt(d)) xi_i.
  The weighted additive contribution for Adam-SHANG is
    (sigma0^2/d) Tr(P_k^{-1}).
  The script records its maximum over the simulated trajectories and
  iterations as an empirical diagnostic, not a deterministic bound.
  Keeping the oracle common across methods avoids an algorithm-dependent
  noise model.

Reproducibility notes:
  - This script uses a synthetic convex benchmark and does not require external data.
  - The NumPy random seed is fixed to 42.
  - Spectral-step comparisons use isolated deterministic seeds, so adding
    them does not change the legacy random stream used by the other methods.
  - NR is the number of Monte Carlo trajectories used to estimate E[F(x_k)].
  - Adam's lr0 is selected from LR_GRID by a pilot grid search of length T_GS.
  - The script saves per-σ₀ convergence plots and final-loss summaries.
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
    ax = np.abs(x)
    return np.sum(np.where(ax <= 1.0, ax**d, 1.0 + d*(ax - 1.0)), axis=-1)

def gradF(x, d):
    ax = np.abs(x)
    return np.where(ax <= 1.0,
                    np.where(ax > 0, d * ax**(d-2) * x, 0.0),
                    d * np.sign(x))

def noisy_g(x, s1, s0, d):
    gf = gradF(x, d)
    if s1 == 0.0 and s0 == 0.0:
        return gf
    Z  = np.random.randn(*x.shape)
    xi = np.random.randn(*x.shape)
    return (1.0 + s1 * Z) * gf + (s0 / np.sqrt(d)) * xi

def project_y(y, R=PROJECTION_RADIUS):
    """Projection onto H_R={y: ||y||_inf <= R/2}."""
    return np.clip(y, -R / 2.0, R / 2.0)

def save_final_loss_table(rows, out_dir):
    """Save rebuttal-ready final-loss summaries as CSV and Markdown."""
    columns = [
        "d", "sigma0", "sigma1", "optimizer", "evaluation_point", "lr0", "final_loss",
        "effective_sigma0_sq",
    ]
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
def sgd(d, s1, s0, T, nr):
    L   = d * (d - 1)
    eta = 1.0 / (L * (1.0 + s1**2))
    x   = np.ones((nr, d))
    res = np.zeros(T)
    res[0] = np.mean(F(x, d))
    for k in range(T - 1):
        x = x - eta * noisy_g(x, s1, s0, d)
        res[k+1] = np.mean(F(x, d))
    return res


#  SHANG
def shang(d, s1, s0, T, nr):
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
        g     = noisy_g(x, s1, s0, d)
        v_new = v - (ap / gp) * g
        x_new = (1.0/(1+alpha))*x + (alpha/(1+alpha))*v_new \
                - beta*(alpha/(1+alpha))*g
        res[k+1] = np.mean(F(x_new, d))
        x, v = x_new, v_new
    return res

#  Adam
def adam_opt(d, s1, s0, T, nr, lr0, b1=0.9, b2=0.999, eps=1e-8, warmup=100):
    x  = np.ones((nr, d))
    m  = np.zeros((nr, d))
    v  = np.full((nr, d), eps)
    res = np.zeros(T)
    res[0] = np.mean(F(x, d))
    for k in range(T - 1):
        lr_k = lr0 * min(k + 1, warmup) / warmup / np.sqrt(k + 1)
        g    = noisy_g(x, s1, s0, d)
        m    = b1*m + (1-b1)*g
        v    = b2*v + (1-b2)*g**2
        mh   = m / (1 - b1**(k+1))
        vh   = v / (1 - b2**(k+1))
        P    = np.sqrt(np.maximum(vh, eps))
        x    = x - lr_k * mh / (P + eps)
        res[k+1] = np.mean(F(x, d))
    return res

# Adam baseline tuning: select lr0 from LR_GRID by minimizing mean+std
# over the last 20% of a pilot run. This documents the baseline
# hyperparameter selection used in the reported convex experiment.
def adam_grid_search(d, s1, s0, T_gs, nr, lrs):
    best_lr, best_val = lrs[0], np.inf
    for lr in lrs:
        r    = adam_opt(d, s1, s0, T_gs, nr, lr)
        tail = r[int(0.8*T_gs):]
        crit = float(np.mean(tail) + np.std(tail))
        if crit < best_val:
            best_val, best_lr = crit, lr
    print(f"      Adam grid → lr₀*={best_lr:.3g}  (mean+std={best_val:.4g})")
    return best_lr

#  Adam-SHANG — fixed R=1 with projection onto H_R
def adam_shang(d, s1, s0, T, nr, eps=1e-8, stepsize_rule="trace",
               metric_floor_stats=None):
    L    = d * (d - 1)
    s1sq = s1**2
    trace_safety_factor = np.sqrt(2.0 / (np.sqrt(d) + 1.0))

    x    = np.ones((nr, d))
    y    = np.zeros((nr, d))

    R    = PROJECTION_RADIUS

    gf0  = gradF(x, d)
    initial_second_moment = (1.0 + s1sq) * gf0**2 + s0**2 / d
    P    = np.sqrt(np.maximum(initial_second_moment, eps))
    if metric_floor_stats is not None:
        init_metric_floor_stats(metric_floor_stats, initial_second_moment, P, eps)
    Plag = P.copy()

    g    = noisy_g(x, s1, s0, d)

    iP      = 1.0 / np.maximum(P, eps)
    max_trace_pinv = float(np.max(np.sum(iP, axis=1)))
    if stepsize_rule == "trace":
        trPinv1 = np.sum(iP,      axis=1, keepdims=True)
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
        max_trace_pinv = max(
            max_trace_pinv, float(np.max(np.sum(iP, axis=1))))
        if stepsize_rule == "trace":
            trPinv1 = np.sum(iP,      axis=1, keepdims=True)
            trPinv2 = np.sum(iP ** 2, axis=1, keepdims=True)
            surrogate = trPinv1 / trPinv2
            alpha = trace_safety_factor * np.sqrt(surrogate / (2.0 * L)) / (1.0 + s1sq)
        else:
            lambda_min = np.min(P, axis=1, keepdims=True)
            alpha = np.sqrt(lambda_min / (2.0 * L)) / (1.0 + s1sq)

        iPlag = 1.0 / np.maximum(Plag, eps)
        x_new = (1.0/(1+alpha))*x + (alpha/(1+alpha))*y \
                - (eta/(1+alpha)) * iPlag * g

        g1 = noisy_g(x_new, s1, s0, d)

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

        Plag, P, x, y, g, eta = P.copy(), P_new, x_new, y_new, g1, eta1

    effective_s0_sq = (s0**2 / d) * max_trace_pinv
    return res, effective_s0_sq

#  Adam-SHANG-s — fixed R=1 with projection onto H_R
def adam_shangs(d, s1, s0, T, nr, eps=1e-8, stepsize_rule="trace",
                metric_floor_stats=None):
    L    = d * (d - 1)
    s1sq = s1**2
    trace_safety_factor = np.sqrt(2.0 / (np.sqrt(d) + 1.0))

    x    = np.ones((nr, d))
    y    = np.zeros((nr, d))

    R    = PROJECTION_RADIUS

    gf0  = gradF(x, d)
    initial_second_moment = (1.0 + s1sq) * gf0**2 + s0**2 / d
    P    = np.sqrt(np.maximum(initial_second_moment, eps))
    if metric_floor_stats is not None:
        init_metric_floor_stats(metric_floor_stats, initial_second_moment, P, eps)

    g    = noisy_g(x, s1, s0, d)

    iP      = 1.0 / np.maximum(P, eps)
    max_trace_pinv = float(np.max(np.sum(iP, axis=1)))
    if stepsize_rule == "trace":
        trPinv1 = np.sum(iP,      axis=1, keepdims=True)
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
        max_trace_pinv = max(
            max_trace_pinv, float(np.max(np.sum(iP, axis=1))))
        if stepsize_rule == "trace":
            trPinv1 = np.sum(iP,      axis=1, keepdims=True)
            trPinv2 = np.sum(iP ** 2, axis=1, keepdims=True)
            surrogate = trPinv1 / trPinv2
            talpha = trace_safety_factor * np.sqrt(surrogate / (2.0 * L)) / (1.0 + s1sq)
        else:
            lambda_min = np.min(P, axis=1, keepdims=True)
            talpha = np.sqrt(lambda_min / (2.0 * L)) / (1.0 + s1sq)
        alpha = talpha /(1.0-talpha)
        x_new = (1.0/(1+alpha))*x + (alpha/(1+alpha))*y \
                - (eta/(1+alpha)) * iP * g

        g1 = noisy_g(x_new, s1, s0, d)

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

        P, x, y, g, eta = P_new, x_new, y_new, g1, eta1

    effective_s0_sq = (s0**2 / d) * max_trace_pinv
    return res, effective_s0_sq

#  Experiment configuration and execution (no experiment runs on import).
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=100000,
                        help="Recorded iterates, including initialization (default: 100000).")
    parser.add_argument("--pilot-steps", type=int, default=30000,
                        help="Recorded iterates per Adam tuning trial (default: 30000).")
    parser.add_argument("--runs", type=int, default=200,
                        help="Monte Carlo trajectories (default: 200).")
    parser.add_argument("--dim", type=int, default=16,
                        help="Problem dimension; default: 16.")
    parser.add_argument("--output", default="./adam_shang_results_sigma0_nonzero",
                        help="Output directory; existing same-name result files are overwritten.")
    args = parser.parse_args(argv)
    if min(args.steps, args.pilot_steps, args.runs) < 1:
        parser.error("--steps, --pilot-steps, and --runs must be positive.")
    if args.dim < 2:
        parser.error("--dim must be at least 2.")
    T_MAIN, T_GS, NR = args.steps, args.pilot_steps, args.runs
    D = args.dim
    OUT_DIR = args.output
    np.random.seed(42)

    S1      = 10
    SIGMA0S = [0.5, 1.0, 3.0]

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
    L  = D * (D - 1)

    print("=" * 65)
    print(f"  sigma0 comparison experiment: d={D}, sigma1={S1}, sigma0 in {SIGMA0S}")
    print("=" * 65)

    ALL = {}
    FINAL_LOSS_ROWS = []
    METRIC_FLOOR_ROWS = []

    for s0 in SIGMA0S:
        nr  = NR
        print(f"\n{'─'*55}\n  d={D}, σ₁={S1}, σ₀={s0}\n{'─'*55}")

        r_sgd = sgd(D, S1, s0, T_MAIN, nr)
        print(f"  SGD          {r_sgd[-1]:.4g}")

        r_shng = shang(D, S1, s0, T_MAIN, nr)
        print(f"  SHANG        {r_shng[-1]:.4g}")

        blr    = adam_grid_search(D, S1, s0, T_GS, nr, LR_GRID)
        r_adam = adam_opt(D, S1, s0, T_MAIN, nr, blr)
        print(f"  Adam         {r_adam[-1]:.4g}  (lr₀={blr:.3g})")

        floor_ashs = {}
        r_ashs, effective_s0_sq_s = adam_shangs(
            D, S1, s0, T_MAIN, nr, metric_floor_stats=floor_ashs)
        print(f"  Adam-SHANG-s (Trace) {r_ashs[-1]:.4g}  "
              f"(observed weighted noise={effective_s0_sq_s:.4g})")

        floor_ash = {}
        r_ash, effective_s0_sq = adam_shang(
            D, S1, s0, T_MAIN, nr, metric_floor_stats=floor_ash)
        print(f"  Adam-SHANG (Trace) {r_ash[-1]:.4g}  "
              f"(observed weighted noise={effective_s0_sq:.4g})")

        spectral_seed = 420000 + int(round(1000 * s0))
        floor_ashs_spectral = {}
        with isolated_numpy_seed(spectral_seed + 1):
            r_ashs_spectral, effective_s0_sq_s_spectral = adam_shangs(
                D, S1, s0, T_MAIN, nr, stepsize_rule="spectral",
                metric_floor_stats=floor_ashs_spectral)
        print(f"  Adam-SHANG-s (spectral) {r_ashs_spectral[-1]:.4g}  "
              f"(observed weighted noise={effective_s0_sq_s_spectral:.4g})")

        floor_ash_spectral = {}
        with isolated_numpy_seed(spectral_seed + 2):
            r_ash_spectral, effective_s0_sq_spectral = adam_shang(
                D, S1, s0, T_MAIN, nr, stepsize_rule="spectral",
                metric_floor_stats=floor_ash_spectral)
        print(f"  Adam-SHANG (spectral)   {r_ash_spectral[-1]:.4g}  "
              f"(observed weighted noise={effective_s0_sq_spectral:.4g})")

        for optimizer, stats in [
                ("Adam-SHANG-s (Trace)", floor_ashs),
                ("Adam-SHANG (Trace)", floor_ash),
                ("Adam-SHANG-s (spectral)", floor_ashs_spectral),
                ("Adam-SHANG (spectral)", floor_ash_spectral)]:
            summary = metric_floor_summary(stats)
            METRIC_FLOOR_ROWS.append({
                "d": D, "sigma0": s0, "sigma1": S1, "optimizer": optimizer,
                **summary,
            })
            print(f"  {optimizer}: min raw P={summary['min_raw_p']}, "
                  f"floor hits={summary['floor_hits']}, "
                  f"first k={summary['first_floor_k']}, "
                  f"floor inactive={summary['floor_inactive']}")

        ALL[s0] = dict(sgd=r_sgd, shang=r_shng, adam=r_adam,
                       ashs=r_ashs,
                       effective_s0_sq_s=effective_s0_sq_s,
                       ash=r_ash,
                       effective_s0_sq=effective_s0_sq,
                       ashs_spectral=r_ashs_spectral,
                       effective_s0_sq_s_spectral=effective_s0_sq_s_spectral,
                       ash_spectral=r_ash_spectral,
                       effective_s0_sq_spectral=effective_s0_sq_spectral,
                       blr=blr)

        results = [
            ("SGD", r_sgd, "", ""),
            ("SHANG", r_shng, "", ""),
            ("Adam", r_adam, f"{blr:.10g}", ""),
            ("Adam-SHANG-s (Trace)", r_ashs, "", f"{effective_s0_sq_s:.10e}"),
            ("Adam-SHANG (Trace)", r_ash, "", f"{effective_s0_sq:.10e}"),
            ("Adam-SHANG-s (spectral)", r_ashs_spectral, "",
             f"{effective_s0_sq_s_spectral:.10e}"),
            ("Adam-SHANG (spectral)", r_ash_spectral, "",
             f"{effective_s0_sq_spectral:.10e}"),
        ]
        for optimizer, losses, lr0, effective_constant in results:
            FINAL_LOSS_ROWS.append({
                "d": D,
                "sigma0": s0,
                "sigma1": S1,
                "optimizer": optimizer,
                "evaluation_point": "x_k^+" if optimizer.startswith("Adam-SHANG") else "x_k",
                "lr0": lr0,
                "final_loss": f"{float(losses[-1]):.10e}",
                "effective_sigma0_sq": effective_constant,
            })


    for s0 in SIGMA0S:
        R  = ALL[s0]
        F0 = max(R['sgd'][0], 1e-15)

        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        ax.set_title(rf"$d={D}$, $\sigma_1={S1}$, $\sigma_0={s0}$", fontweight="bold")

        ax.loglog(xs, np.maximum(R['sgd'],   1e-15), color=C["SGD"],          ls="-",  lw=1.8, label="SGD")
        ax.loglog(xs, np.maximum(R['shang'], 1e-15), color=C["SHANG"],        ls="--", lw=1.8, label="SHANG")
        ax.loglog(xs, np.maximum(R['adam'],  1e-15), color=C["Adam"],         ls="-.", lw=1.8, label=rf"Adam $\ell_0$={R['blr']:.2g}")
        ax.loglog(xs, np.maximum(R['ashs'],  1e-15), color=C["Adam-SHANG-s (Trace)"], ls="-",  lw=2.4, label="Adam-SHANG-s (Trace)")
        ax.loglog(xs, np.maximum(R['ash'],   1e-15), color=C["Adam-SHANG (Trace)"],   ls="-",  lw=2.4, label="Adam-SHANG (Trace)")
        ax.loglog(xs, np.maximum(R['ashs_spectral'], 1e-15),
                  color=C["Adam-SHANG-s (spectral)"], ls="--", lw=2.0,
                  label="Adam-SHANG-s (spectral)")
        ax.loglog(xs, np.maximum(R['ash_spectral'], 1e-15),
                  color=C["Adam-SHANG (spectral)"], ls="--", lw=2.0,
                  label="Adam-SHANG (spectral)")
        ax.loglog(xs, F0/xs,      color="grey", ls=":",           lw=0.9, alpha=0.5, label=r"$O(1/k)$")
        ax.loglog(xs, F0/xs**2,   color="grey", ls=(0,(3,1,1,1)), lw=0.9, alpha=0.5, label=r"$O(1/k^2)$")

        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel(r"$\mathbb{E}[f(x_k^+)]$")
        ax.legend(loc="lower left", framealpha=0.92)
        ax.grid(True, which="both", alpha=0.22, ls=":")
        y_top = float(F0) * 2.0
        y_bot = max(min(R['sgd'][-1], R['shang'][-1], R['adam'][-1],
                        R['ashs'][-1], R['ash'][-1],
                        R['ashs_spectral'][-1], R['ash_spectral'][-1]) / 100.0,
                    y_top * 1e-8)
        ax.set_ylim(bottom=y_bot, top=y_top)
        fig.tight_layout()
        fname = f"d{D}_s1{S1}_s0{str(s0).replace('.','p')}.png"
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {fname}")

    save_final_loss_table(FINAL_LOSS_ROWS, OUT_DIR)
    save_metric_floor_table(METRIC_FLOOR_ROWS, OUT_DIR)

    print(f"\nDone. Results saved in {OUT_DIR}/")


if __name__ == "__main__":
    main()
