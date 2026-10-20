"""Stochastic online Reddi experiment; edit SETTINGS or use budget CLI options.

Each optimizer has its own complete recurrence. At round t it plays x_t
BEFORE observing the shared g_t (1010 with probability .01, else -10).
Exact hindsight regret is sum(g_s*x_s) + abs(sum(g_s)).

Default fixed parameters and random streams preserve the corrected online
experiment. Both optional pilot searches are OFF by default. No work runs
on import. Numba only accelerates the loops; .py_func remains available.
"""

import argparse
import csv
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(**_options):
        def decorate(function):
            function.py_func = function
            return function
        return decorate


# SETTINGS -------------------------------------------------------------------
N_FINAL_STOCH = 10_000_000
NR_STOCH = 30
N_CHECKPOINTS = 10_000
SEED = 0
SAVE_PLOTS = True
OUTPUT_DIR = None  # None creates a fresh timestamped directory.

ETA_BASELINE = 0.01
BETA1, BETA2 = 0.9, 0.99
LAM, BETA, GAMMA = 0.001, 0.0001, 0.05
P_MIN, EPS = 1e-6, 1e-8

SEARCH_BASELINES = False  # True searches Adam and AMSGrad independently.
SEARCH_SHANG = False  # True searches lagged SHANG; sync uses the same tuple.
N_SEARCH = 50_000
N_SEARCH_RUNS = 3
eta_baseline_grid = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]
lam_grid = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]
beta_grid = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]
gamma_grid = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]

METHODS = ("Adam", "AMSGrad", "Adam-SHANG-s", "Adam-SHANG")
METRIC_METHODS = METHODS[2:]
ACTION_KEYS = ("played_actions", "average_regrets", "cumulative_losses",
               "cumulative_gradients", "next_actions")


@njit(cache=False)
def check_inputs(gradients, checkpoints):
    """Checkpoints are one-based played-action rounds."""
    if (len(gradients) < 1 or len(checkpoints) < 1 or checkpoints[0] < 1
            or checkpoints[-1] != len(gradients)):
        raise ValueError("Checkpoints must lie in [1,T] and include T")
    for j in range(1, len(checkpoints)):
        if checkpoints[j] <= checkpoints[j - 1]:
            raise ValueError("Checkpoints must be strictly increasing")


# Adam -----------------------------------------------------------------------
@njit(cache=False)
def run_adam(gradients, checkpoints, eta=0.01, beta1=0.9, beta2=0.99, eps=1e-8):
    """Constant stepsize, bias-corrected Adam; return 1-D arrays."""
    check_inputs(gradients, checkpoints)
    if eta <= 0 or eps <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("Invalid Adam parameters")
    count = len(checkpoints)
    played = np.empty(count)
    regrets = np.empty(count)
    losses = np.empty(count)
    sums = np.empty(count)
    following = np.empty(count)
    x = 0.0
    m = 0.0
    v = 0.0
    cumulative_loss = 0.0
    cumulative_gradient = 0.0
    record = 0

    for k in range(len(gradients)):
        t = k + 1
        g = gradients[k]
        if not np.isfinite(g):
            raise ValueError("Observed gradients must be finite")
        save = t == checkpoints[record]
        cumulative_gradient += g
        cumulative_loss += g * x  # x_t was chosen before observing g_t.
        if save:
            played[record] = x
            losses[record] = cumulative_loss
            sums[record] = cumulative_gradient
            regrets[record] = (cumulative_loss + abs(cumulative_gradient)) / t

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        mhat = m / (1.0 - beta1**t)
        vhat = v / (1.0 - beta2**t)
        x = min(1.0, max(-1.0, x - eta * mhat / (np.sqrt(vhat) + eps)))
        if save:
            following[record] = x
            record += 1

    return {"played_actions": played, "average_regrets": regrets,
            "cumulative_losses": losses, "cumulative_gradients": sums,
            "next_actions": following}


# AMSGrad --------------------------------------------------------------------
@njit(cache=False)
def run_amsgrad(gradients, checkpoints, eta=0.01, beta1=0.9, beta2=0.99, eps=1e-8):
    """Constant stepsize; retain the original no-bias-correction convention."""
    check_inputs(gradients, checkpoints)
    if eta <= 0 or eps <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("Invalid AMSGrad parameters")
    count = len(checkpoints)
    played = np.empty(count)
    regrets = np.empty(count)
    losses = np.empty(count)
    sums = np.empty(count)
    following = np.empty(count)
    x = 0.0
    m = 0.0
    v = 0.0
    vmax = 0.0
    cumulative_loss = 0.0
    cumulative_gradient = 0.0
    record = 0

    for k in range(len(gradients)):
        t = k + 1
        g = gradients[k]
        if not np.isfinite(g):
            raise ValueError("Observed gradients must be finite")
        save = t == checkpoints[record]
        cumulative_gradient += g
        cumulative_loss += g * x
        if save:
            played[record] = x
            losses[record] = cumulative_loss
            sums[record] = cumulative_gradient
            regrets[record] = (cumulative_loss + abs(cumulative_gradient)) / t

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        vmax = max(vmax, v)
        x = min(1.0, max(-1.0, x - eta * m / (np.sqrt(vmax) + eps)))
        if save:
            following[record] = x
            record += 1

    return {"played_actions": played, "average_regrets": regrets,
            "cumulative_losses": losses, "cumulative_gradients": sums,
            "next_actions": following}


# Adam-SHANG: lagged preconditioner -------------------------------------------
@njit(cache=False)
def run_adam_shang(gradients, g0, checkpoints, lam=0.001, beta=0.0001,
                   gamma=0.05, p_min=1e-6, eps=1e-8):
    """Unprojected y; old metric in y/correction; current-alpha correction."""
    check_inputs(gradients, checkpoints)
    if (lam <= 0 or beta < 0 or gamma <= 0 or p_min <= 0 or eps <= 0
            or not np.isfinite(g0)):
        raise ValueError("Invalid Adam-SHANG parameters")
    count = len(checkpoints)
    played = np.empty(count)
    regrets = np.empty(count)
    losses = np.empty(count)
    sums = np.empty(count)
    following = np.empty(count)
    metrics = np.empty(count)
    auxiliary = np.empty(count)
    x = 0.0
    y = 0.0
    P_current = max(gamma * g0 * g0 + p_min, p_min)
    minimum_metric = P_current
    maximum_abs_y = 0.0
    cumulative_loss = 0.0
    cumulative_gradient = 0.0
    record = 0

    for k in range(len(gradients)):
        t = k + 1
        g = gradients[k]
        if not np.isfinite(g):
            raise ValueError("Observed gradients must be finite")
        save = t == checkpoints[record]
        cumulative_gradient += g
        cumulative_loss += g * x
        if save:
            played[record] = x
            losses[record] = cumulative_loss
            sums[record] = cumulative_gradient
            regrets[record] = (cumulative_loss + abs(cumulative_gradient)) / t

        P_previous = max(P_current, eps)  # P_{t-1}
        alpha_previous = lam * np.sqrt(P_previous)  # alpha_{t-1}
        P_current = max((P_previous + alpha_previous * gamma * g * g / P_previous)
                        / (1.0 + alpha_previous), p_min)
        y -= alpha_previous * g / P_previous  # y_t: old P, full alpha.
        alpha_current = lam * np.sqrt(max(P_current, eps))  # alpha_t
        # Reuse g_t, with P_{t-1}; choose x_{t+1} for the next round.
        x = min(1.0, max(-1.0, (
            x + alpha_current * y - alpha_current * beta * g / P_previous)
            / (1.0 + alpha_current)))

        if not np.isfinite(P_current) or not np.isfinite(y):
            raise FloatingPointError("Non-finite metric or auxiliary state")
        minimum_metric = min(minimum_metric, P_current)
        maximum_abs_y = max(maximum_abs_y, abs(y))
        if save:
            following[record] = x
            metrics[record] = P_current
            auxiliary[record] = y
            record += 1

    return {"played_actions": played, "average_regrets": regrets,
            "cumulative_losses": losses, "cumulative_gradients": sums,
            "next_actions": following, "metrics": metrics, "auxiliary_states": auxiliary,
            "minimum_metrics": np.array([minimum_metric]),
            "maximum_abs_auxiliary": np.array([maximum_abs_y])}


# Adam-SHANG-s: synchronous preconditioner ------------------------------------
@njit(cache=False)
def run_adam_shangs(gradients, g0, checkpoints, lam=0.001, beta=0.0001,
                    gamma=0.05, p_min=1e-6, eps=1e-8):
    """Positive-root metric; new metric in y/correction; no projection of y."""
    check_inputs(gradients, checkpoints)
    if (lam <= 0 or beta < 0 or gamma <= 0 or p_min <= 0 or eps <= 0
            or not np.isfinite(g0)):
        raise ValueError("Invalid Adam-SHANG-s parameters")
    count = len(checkpoints)
    played = np.empty(count)
    regrets = np.empty(count)
    losses = np.empty(count)
    sums = np.empty(count)
    following = np.empty(count)
    metrics = np.empty(count)
    auxiliary = np.empty(count)
    x = 0.0
    y = 0.0
    P_current = max(gamma * g0 * g0 + p_min, p_min)
    minimum_metric = P_current
    maximum_abs_y = 0.0
    cumulative_loss = 0.0
    cumulative_gradient = 0.0
    record = 0

    for k in range(len(gradients)):
        t = k + 1
        g = gradients[k]
        if not np.isfinite(g):
            raise ValueError("Observed gradients must be finite")
        save = t == checkpoints[record]
        cumulative_gradient += g
        cumulative_loss += g * x
        if save:
            played[record] = x
            losses[record] = cumulative_loss
            sums[record] = cumulative_gradient
            regrets[record] = (cumulative_loss + abs(cumulative_gradient)) / t

        P_previous = max(P_current, eps)  # P_{t-1}
        alpha_previous = lam * np.sqrt(P_previous)
        tilde_alpha = alpha_previous / (1.0 + alpha_previous)
        shrink = P_previous / (1.0 + alpha_previous)
        # Positive root: P_t^2 - shrink*P_t - tilde_alpha*gamma*g_t^2 = 0.
        # hypot evaluates sqrt(shrink^2 + 4*tilde_alpha*gamma*g_t^2) stably.
        P_current = max(0.5 * (shrink + np.hypot(
            shrink, 2.0 * np.sqrt(tilde_alpha * gamma) * abs(g))), p_min)
        denominator = max(P_current, eps)
        y -= tilde_alpha * g / denominator  # y_t: new P, tilde_alpha.
        alpha_current = lam * np.sqrt(denominator)
        # Reuse g_t, now with P_t; choose x_{t+1} for the next round.
        x = min(1.0, max(-1.0, (
            x + alpha_current * y - alpha_current * beta * g / denominator)
            / (1.0 + alpha_current)))

        if not np.isfinite(P_current) or not np.isfinite(y):
            raise FloatingPointError("Non-finite metric or auxiliary state")
        minimum_metric = min(minimum_metric, P_current)
        maximum_abs_y = max(maximum_abs_y, abs(y))
        if save:
            following[record] = x
            metrics[record] = P_current
            auxiliary[record] = y
            record += 1

    return {"played_actions": played, "average_regrets": regrets,
            "cumulative_losses": losses, "cumulative_gradients": sums,
            "next_actions": following, "metrics": metrics, "auxiliary_states": auxiliary,
            "minimum_metrics": np.array([minimum_metric]),
            "maximum_abs_auxiliary": np.array([maximum_abs_y])}


# Samples, pilot scoring, and checkpoints ------------------------------------
def sample_gradients(steps, run, stream=1, seed=SEED):
    rng = np.random.default_rng(np.random.SeedSequence([seed, run, stream]))
    return np.where(rng.random(steps) < 0.01, 1010.0, -10.0)


def sample_initial_gradient(run, stream=0, seed=SEED):
    rng = np.random.default_rng(np.random.SeedSequence([seed, run, stream]))
    return 1010.0 if rng.random() < 0.01 else -10.0


def make_checkpoints(steps, count):
    return np.unique(np.r_[1, np.geomspace(1, steps, min(steps, count)).astype(np.int64), steps])


def score(result):
    """Use every pilot round, not log checkpoints; lower is better."""
    tail = result["average_regrets"][int(0.8 * len(result["average_regrets"])):]
    return (float(np.mean(tail)) + float(np.std(tail))
            + abs(float(result["next_actions"][-1]) + 1.0))


# Saving and plotting --------------------------------------------------------
def save_results(outdir, checkpoints, results, initial_gradients, settings, search_rows):
    """Keep the existing NPZ fields and method/metric column ordering."""
    played = np.stack([results[name]["played_actions"] for name in METHODS], axis=1)
    regrets = np.stack([results[name]["average_regrets"] for name in METHODS], axis=1)
    losses = np.stack([results[name]["cumulative_losses"] for name in METHODS], axis=1)
    following = np.stack([results[name]["next_actions"] for name in METHODS], axis=1)
    sums = results["Adam"]["cumulative_gradients"]
    metrics = np.stack([results[name]["metrics"] for name in METRIC_METHODS], axis=1)
    auxiliary = np.stack([results[name]["auxiliary_states"] for name in METRIC_METHODS], axis=1)
    minimum_metrics = np.stack([results[name]["minimum_metrics"] for name in METRIC_METHODS])
    maximum_abs_auxiliary = np.stack([results[name]["maximum_abs_auxiliary"] for name in METRIC_METHODS])

    expected = (losses + np.abs(sums)[:, None, :]) / checkpoints[:, None, None]
    if not np.array_equal(regrets, expected):
        raise AssertionError("Saved regret does not match the hindsight formula")
    for name in METHODS:
        if not np.array_equal(results[name]["cumulative_gradients"], sums):
            raise AssertionError("Methods did not use the same oracle stream")
    if (not np.all(np.isfinite(played)) or not np.all(np.isfinite(regrets))
            or not np.all(np.isfinite(following)) or np.any(np.abs(played) > 1.0)
            or np.any(np.abs(following) > 1.0) or np.any(metrics < settings["p_min"])):
        raise FloatingPointError("Saved states failed finiteness or feasibility checks")

    np.savez_compressed(outdir / "trajectories.npz", rounds=checkpoints,
                        methods=np.array(METHODS), metric_methods=np.array(METRIC_METHODS),
                        played_actions=played, average_regrets=regrets,
                        cumulative_losses=losses, cumulative_gradients=sums,
                        next_actions=following, metrics=metrics, auxiliary_states=auxiliary,
                        minimum_metrics=minimum_metrics, maximum_abs_auxiliary=maximum_abs_auxiliary,
                        initial_gradients=initial_gradients)
    rows = [{"method": name, "run": run,
             "played_x_T": float(played[-1, j, run]),
             "next_x_Tplus1": float(following[-1, j, run]),
             "average_regret": float(regrets[-1, j, run]),
             "cumulative_loss": float(losses[-1, j, run]),
             "cumulative_gradient": float(sums[-1, run])}
            for j, name in enumerate(METHODS) for run in range(settings["runs"])]
    with (outdir / "final_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if search_rows:
        with (outdir / "grid_search.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=search_rows[0].keys())
            writer.writeheader()
            writer.writerows(search_rows)
    with (outdir / "settings.json").open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2, allow_nan=False)


def save_fig(results, key, ylabel, stem, rounds, outdir):
    """Pointwise medians with the existing online figure style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 17,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 12})
    colors = ("green", "blue", "orange", "red")
    styles = ((0, ()), (0, (6, 2)), (0, (2, 1)), (0, (4, 1, 1, 1)))
    widths = (2.0, 2.0, 2.0, 2.5)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for j, name in enumerate(METHODS):
        ax.plot(rounds, np.median(results[name][key], axis=-1), color=colors[j],
                linestyle=styles[j], linewidth=widths[j], label=name)
    if key == "played_actions":
        ax.axhline(-1.0, color="black", lw=1.4, ls="--",
                   label=r"$x^\star=-1$", alpha=0.85)
        ax.set_ylim(-1.2, 1.2)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.legend(framealpha=0.9, loc="lower left")
    else:
        ax.axhline(0.0, color="black", lw=0.8, ls=":", alpha=0.6)
        ax.legend(framealpha=0.9)
    ax.set_xscale("log")
    ax.set_xlim(1, max(2, int(rounds[-1])))
    ax.set_xlabel("Steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(outdir / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=N_FINAL_STOCH,
                        help="Online rounds per trajectory (default: 10000000).")
    parser.add_argument("--runs", type=int, default=NR_STOCH,
                        help="Independent evaluation trajectories (default: 30).")
    parser.add_argument("--checkpoints", type=int, default=N_CHECKPOINTS,
                        help="Requested log-spaced checkpoints (default: 10000).")
    parser.add_argument("--pilot-steps", type=int, default=N_SEARCH,
                        help="Rounds per optional pilot trial; does not enable searching.")
    parser.add_argument("--pilot-runs", type=int, default=N_SEARCH_RUNS,
                        help="Trajectories per optional pilot trial; does not enable searching.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="A new output directory; defaults to a timestamped directory beside this script.")
    args = parser.parse_args(argv)
    n_final_stoch, nr_stoch, n_checkpoints = args.steps, args.runs, args.checkpoints
    n_search, n_search_runs, output_dir = args.pilot_steps, args.pilot_runs, args.output
    if min(n_final_stoch, nr_stoch, n_checkpoints, n_search, n_search_runs) < 1 or SEED < 0:
        raise ValueError("Iteration/run counts must be positive and SEED nonnegative")
    outdir = (Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent /
              f"reddi_online_stochastic_{datetime.now():%Y%m%d_%H%M%S_%f}")
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    best_adam_eta = ETA_BASELINE
    best_ams_eta = ETA_BASELINE
    best_shang = {"lam": LAM, "beta": BETA, "gamma": GAMMA}
    search_rows = []
    if not NUMBA_AVAILABLE:
        print("Numba unavailable: the same recurrence loops run in Python.", flush=True)

    # Every candidate shares independent pilot streams:
    # stream 2 contains pilot gradients; stream 3 supplies independent g0.
    if SEARCH_BASELINES or SEARCH_SHANG:
        pilot_checkpoints = np.arange(1, n_search + 1, dtype=np.int64)
        pilot_samples = [(sample_initial_gradient(run, stream=3, seed=SEED),
                          sample_gradients(n_search, run, stream=2, seed=SEED))
                         for run in range(n_search_runs)]

    # Separate Adam learning-rate search.
    if SEARCH_BASELINES:
        print("Searching Adam base learning rate on pilot runs...", flush=True)
        best_score = float("inf")
        for eta in eta_baseline_grid:
            candidate_score = float(np.mean([
                score(run_adam(g, pilot_checkpoints, eta, BETA1, BETA2, EPS))
                for _, g in pilot_samples]))
            search_rows.append({"method": "Adam", "eta": eta, "lam": None,
                                "beta": None, "gamma": None, "score": candidate_score})
            if candidate_score < best_score:
                best_score, best_adam_eta = candidate_score, eta

        # AMSGrad does not inherit Adam's selected learning rate.
        print("Searching AMSGrad base learning rate on pilot runs...", flush=True)
        best_score = float("inf")
        for eta in eta_baseline_grid:
            candidate_score = float(np.mean([
                score(run_amsgrad(g, pilot_checkpoints, eta, BETA1, BETA2, EPS))
                for _, g in pilot_samples]))
            search_rows.append({"method": "AMSGrad", "eta": eta, "lam": None,
                                "beta": None, "gamma": None, "score": candidate_score})
            if candidate_score < best_score:
                best_score, best_ams_eta = candidate_score, eta
    else:
        print(f"Baseline search OFF: Adam/AMSGrad keep eta={ETA_BASELINE:g}.", flush=True)

    # Optional lagged SHANG search; synchronous SHANG uses the same chosen tuple.
    if SEARCH_SHANG:
        print("Searching lagged Adam-SHANG parameters on pilot runs...", flush=True)
        best_score = float("inf")
        for lam, beta, gamma in product(lam_grid, beta_grid, gamma_grid):
            candidate_score = float(np.mean([
                score(run_adam_shang(g, g0, pilot_checkpoints, lam, beta, gamma, P_MIN, EPS))
                for g0, g in pilot_samples]))
            search_rows.append({"method": "Adam-SHANG", "eta": None, "lam": lam,
                                "beta": beta, "gamma": gamma, "score": candidate_score})
            if candidate_score < best_score:
                best_score = candidate_score
                best_shang = {"lam": lam, "beta": beta, "gamma": gamma}
    else:
        print(f"SHANG search OFF: keeping {best_shang}.", flush=True)

    # Stochastic evaluation: one common oracle vector per independent run.
    checkpoints = make_checkpoints(n_final_stoch, n_checkpoints)
    stoch_raw = {name: {key: np.empty((len(checkpoints), nr_stoch)) for key in ACTION_KEYS}
                 for name in METHODS}
    for name in METRIC_METHODS:
        stoch_raw[name]["metrics"] = np.empty((len(checkpoints), nr_stoch))
        stoch_raw[name]["auxiliary_states"] = np.empty((len(checkpoints), nr_stoch))
        stoch_raw[name]["minimum_metrics"] = np.empty(nr_stoch)
        stoch_raw[name]["maximum_abs_auxiliary"] = np.empty(nr_stoch)
    initial_gradients = np.empty(nr_stoch)
    run_records = []
    started = time.monotonic()
    print(f"Stochastic evaluation: {n_final_stoch:,} rounds x {nr_stoch} runs.", flush=True)

    for run in range(nr_stoch):
        run_started = time.monotonic()
        g0 = sample_initial_gradient(run, stream=0, seed=SEED)
        gradients = sample_gradients(n_final_stoch, run, stream=1, seed=SEED)
        initial_gradients[run] = g0
        trial = {
            "Adam": run_adam(gradients, checkpoints, best_adam_eta, BETA1, BETA2, EPS),
            "AMSGrad": run_amsgrad(gradients, checkpoints, best_ams_eta, BETA1, BETA2, EPS),
            "Adam-SHANG-s": run_adam_shangs(gradients, g0, checkpoints,
                best_shang["lam"], best_shang["beta"], best_shang["gamma"], P_MIN, EPS),
            "Adam-SHANG": run_adam_shang(gradients, g0, checkpoints,
                best_shang["lam"], best_shang["beta"], best_shang["gamma"], P_MIN, EPS),
        }
        for name in METHODS:
            for key in ACTION_KEYS:
                stoch_raw[name][key][:, run] = trial[name][key]
        for name in METRIC_METHODS:
            for key in ("metrics", "auxiliary_states"):
                stoch_raw[name][key][:, run] = trial[name][key]
            for key in ("minimum_metrics", "maximum_abs_auxiliary"):
                stoch_raw[name][key][run] = trial[name][key][0]
        run_records.append({"run": run, "initial_gradient": g0,
            "initial_metric": max(best_shang["gamma"] * g0**2 + P_MIN, P_MIN),
            "gradient_sha256": hashlib.sha256(gradients.view(np.uint8)).hexdigest(),
            "rare_gradient_count": int(np.count_nonzero(gradients == 1010.0)),
            "elapsed_seconds": time.monotonic() - run_started})
        del gradients, trial
        print(f"  run {run + 1}/{nr_stoch}: {run_records[-1]['elapsed_seconds']:.2f}s", flush=True)

    settings = {
        "steps": n_final_stoch, "runs": nr_stoch, "seed": SEED,
        "checkpoints": n_checkpoints, **best_shang, "eta": ETA_BASELINE,
        "selected_baseline_eta": {"Adam": best_adam_eta, "AMSGrad": best_ams_eta},
        "beta1": BETA1, "beta2": BETA2, "p_min": P_MIN, "eps": EPS,
        "stochastic": True, "methods": METHODS, "metric_methods": METRIC_METHODS,
        "gradient_values": [1010.0, -10.0], "gradient_probabilities": [0.01, 0.99],
        "mean_gradient": 0.2,
        "seed_construction": "SeedSequence([seed, run, stream]); 0=g0, 1=evaluation, 2=pilot, 3=pilot g0",
        "gradient_hash_format": "SHA256 of contiguous float64 gradient vector bytes",
        "sampling": "one fresh common draw per round; independent initialization sample",
        "initialization": "x1=y0=0; P0=max(gamma*g0^2+p_min,p_min); no bootstrap correction",
        "action_timing": "played x_t precedes g_t; next_actions stores x_{t+1}",
        "regret": "R_t/t=(sum_{s<=t} g_s*x_s + abs(sum_{s<=t} g_s))/t",
        "comparator": "best fixed action in [-1,1] in hindsight at each checkpoint",
        "adam_bias_correction": True, "amsgrad_bias_correction": False,
        "baseline_learning_rate": "constant selected eta; no decay",
        "parameter_search": SEARCH_BASELINES or SEARCH_SHANG,
        "search_baselines": SEARCH_BASELINES, "search_shang": SEARCH_SHANG,
        "pilot_steps": n_search, "pilot_runs": n_search_runs,
        "pilot_score": "per-run mean+std of final20% uniformly sampled per-round average regret, plus abs(final next_action+1); then mean across runs",
        "search_grids": {"baseline_eta": eta_baseline_grid, "lam": lam_grid,
                         "beta": beta_grid, "gamma": gamma_grid},
        "search_tie_rule": "first candidate in the declared grid order",
        "shang_correction": "current alpha_t*beta; lagged P_{t-1}, synchronous P_t denominator",
        "metric_floor": "p_min at initialization and after every metric update",
        "denominator_safeguard": "max(P,eps)", "auxiliary_projection": False,
        "primal_projection": "coordinate clipping onto [-1,1] after each update",
        "summary": "pointwise medians over independent runs only",
        "checkpoint_policy": "unique logarithmically spaced one-based rounds, including 1 and T",
        "numba_available": NUMBA_AVAILABLE, "numpy_version": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_records": run_records, "elapsed_seconds": time.monotonic() - started,
    }
    save_results(outdir, checkpoints, stoch_raw, initial_gradients, settings, search_rows)
    if SAVE_PLOTS:
        save_fig(stoch_raw, "average_regrets", rf"$R_t/t$ (median, {nr_stoch} runs)",
                 "reddi_online_stoch_median_regret", checkpoints, outdir)
        save_fig(stoch_raw, "played_actions", rf"$x_t$ (median, {nr_stoch} runs)",
                 "reddi_online_stoch_median_iterates", checkpoints, outdir)
    for name in METHODS:
        print(f"{name:14s}: median played x_T={np.median(stoch_raw[name]['played_actions'][-1]):.8g}, "
              f"median R_T/T={np.median(stoch_raw[name]['average_regrets'][-1]):.8g}")
    print(f"Saved to {outdir}")


if __name__ == "__main__":
    main()
