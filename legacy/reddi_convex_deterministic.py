"""Stochastic constrained convex stability test with theory-derived coupling.

This legacy module name is retained for compatibility. Both this file and
reddi_convex_stochastic.py now run only the iid stochastic experiment.

Minimize f(x)=x over [-1,1], with x*=-1. The Reddi-family toy uses gradients
n^2 with probability 1/n and -n otherwise (default n=50), so E[g]=1 and
E[g^2]=n^3+n^2-n.

The SHANG variants retain the theory-derived spectral parameter couplings
with the valid smoothness bound L=1, R=2 and P0=sqrt(E[g^2]). The auxiliary
y and corrected xplus are clipped to [-1,1]; their convex-combination query
is therefore feasible. This additional primal projection makes the test
a constrained stability comparison, not the unconstrained theorem itself.
There is no fresh SHANG initialization sample or artificial metric floor.
The reported outputs are corrected xplus for the SHANG methods and the
usual updated x for Adam/AMSGrad. The sampled (uncorrected) SHANG x is
also saved, making the distinction explicit and inspectable.

Adam/AMSGrad independently tune lr0 on separate pilot trajectories using
the last-20%-of-pilot mean-objective mean+std criterion; exact ties are
resolved by the smaller full-pilot mean gap. Both use fixed
betas=(0.9,0.999), eps=1e-8, lr0/sqrt(k+1), no warmup or weight decay.
AMSGrad takes the running maximum of the raw second-moment EMA, then
applies the same bias correction as Adam. Pilot/final seeds are separate;
all methods and learning-rate candidates share component draws within
each stage (common random numbers), with independent trajectories.

All methods project their primal updates onto the same feasible interval.

Outputs: PNG/PDF figures, per-run checkpoint data (.npz), final-run values,
learning-rate search scores (.csv), and settings/provenance (.json).
Only pointwise stochastic MEDIANS are plotted.
Checkpoint 0 is initialization; --steps counts actual updates.
Use --replot-from RUN_DIRECTORY to select the four active methods from
saved stochastic trajectories without rerunning or changing their values.
The new settings record the source hashes and subset-only provenance.

Examples:
  python reddi_convex_stochastic.py --steps 10000000 --runs 30
  python reddi_convex_stochastic.py --replot-from RUN_DIRECTORY --output NEW_DIRECTORY
"""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

LR_GRID = (1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0)
METHODS = ("Adam", "AMSGrad", "Adam-SHANG-s", "Adam-SHANG")
STYLES = {"Adam": ("green", "-"), "AMSGrad": ("blue", "--"),
          "Adam-SHANG-s": ("orange", ":"), "Adam-SHANG": ("red", "-.")}
X_STAR = -1.0
SMOOTHNESS = 1.0  # Any positive Lipschitz bound is valid for f'(x)=1.


@dataclass(frozen=True)
class Config:
    steps: int = 10000000
    runs: int = 30
    pilot_steps: int = 30000
    pilot_runs: int = 30
    seed: int = 20260927
    radius: float = 2.0
    noise_level: int = 50
    x0: float = 0.0
    y0: float = 0.0
    checkpoints: int = 2500

    @property
    def moment_factor(self):
        n = self.noise_level
        return float(n**3 + n**2 - n)

    @property
    def p0(self):
        # Deterministic RMS oracle at x0; no fresh initialization sample.
        return np.sqrt(self.moment_factor)

    def validate(self):
        if self.steps < 1 or not 1 <= self.pilot_steps <= self.steps:
            raise ValueError("Require 1 <= pilot_steps <= steps")
        if self.runs < 1 or self.pilot_runs < 1 or self.checkpoints < 2:
            raise ValueError("Positive run counts and >=2 checkpoints are required")
        if not isinstance(self.noise_level, (int, np.integer)) or self.noise_level < 1:
            raise ValueError("noise_level must be a positive integer")
        if self.radius != 2.0:
            raise ValueError("This constrained benchmark fixes R=2, with y in [-1,1]")
        if not np.isfinite(self.x0) or abs(self.x0) > 1.0:
            raise ValueError("x0 must belong to [-1,1]")
        if abs(self.y0) > self.radius / 2 or not np.isfinite(self.y0):
            raise ValueError("y0 must belong to the projection interval")
        if not np.isfinite(self.p0) or self.p0 <= 0:
            raise ValueError("The RMS initialization must be finite and positive")


def full_gradient(x):
    return np.ones_like(np.asarray(x, dtype=float))


def objective_gap(x):
    """Exact constrained objective gap f(x)-f(-1), for feasible x."""
    x = np.asarray(x, dtype=float)
    return x + 1.0


def oracle_multiplier(rng, runs, noise_level=50):
    n = float(noise_level)
    return np.where(rng.random(runs) < 1.0 / n, n**2, -n)


def baseline_step(x, m, v, vmax, gradient, k, lr, is_amsgrad):
    """Bias-corrected Adam/AMSGrad; k is the one-based update count."""
    m = 0.9 * m + 0.1 * gradient
    v = 0.999 * v + 0.001 * gradient**2
    vmax = np.maximum(vmax, v)
    second_moment = np.where(is_amsgrad, vmax, v)
    direction = (m / (1.0 - 0.9**k)) / (
        np.sqrt(second_moment / (1.0 - 0.999**k)) + 1e-8)
    return np.clip(x - (lr / np.sqrt(k)) * direction, -1.0, 1.0), m, v, vmax


def shang_step(xplus, y, p, multiplier, moment_factor, radius, synchronous=False):
    """Theory-derived spectral coupling with primal and auxiliary projection."""
    alpha = np.sqrt(p / ((6.0 if synchronous else 2.0) * SMOOTHNESS)) / moment_factor
    x = (xplus + alpha * y) / (1.0 + alpha)
    g = multiplier * full_gradient(x)
    if synchronous:
        talpha = alpha / (1.0 + alpha)
        gamma = talpha / (2.0 * radius**2)
        shrink = p / (1.0 + alpha)
        # hypot avoids squaring the entire discriminant explicitly.
        pnext = 0.5 * (shrink + np.hypot(
            shrink, 2.0 * np.sqrt(talpha * gamma) * np.abs(g)))
        y = np.clip(y - talpha * g / pnext, -radius / 2.0, radius / 2.0)
        eta = 3.0 * talpha**2 * moment_factor
        xplus = x - eta * g / pnext
    else:
        gamma = alpha / radius**2
        eta = 2.0 * alpha**2 * moment_factor
        y = np.clip(y - alpha * g / p, -radius / 2.0, radius / 2.0)
        xplus = x - eta * g / p
        pnext = (p + alpha * gamma * g**2 / p) / (1.0 + alpha)
    return np.clip(xplus, -1.0, 1.0), y, pnext, x


def tune_baselines(config):
    """Tune both baselines together using identical independent pilot draws."""
    grid = np.asarray(LR_GRID)
    shape = (2, len(grid), config.pilot_runs)
    x = np.full(shape, config.x0)
    m, v, vmax = (np.zeros(shape) for _ in range(3))
    lr = grid[None, :, None]
    ams = np.array([False, True])[:, None, None]
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0]))
    total = np.zeros((2, len(grid)))
    total_sq = np.zeros_like(total)
    full_total = np.zeros_like(total)
    count = 0
    for k in range(1, config.pilot_steps + 1):
        multiplier = oracle_multiplier(rng, config.pilot_runs, config.noise_level)
        x, m, v, vmax = baseline_step(x, m, v, vmax,
            multiplier * full_gradient(x), k, lr, ams)
        mean_gap = np.mean(objective_gap(x), axis=-1)
        full_total += mean_gap
        if k > int(0.8 * config.pilot_steps):
            total += mean_gap
            total_sq += mean_gap**2
            count += 1
    mean = total / count
    score = mean + np.sqrt(np.maximum(total_sq / count - mean**2, 0.0))
    if not np.all(np.isfinite(score)):
        raise FloatingPointError("Non-finite pilot scores; inspect the candidate trajectories")
    full_mean = full_total / config.pilot_steps
    # A zero terminal gap can tie several converged candidates. Compare
    # their full-pilot error rather than favoring grid order or a slow lr.
    best = [min(range(len(grid)), key=lambda i: (score[j, i], full_mean[j, i], i))
            for j in range(2)]
    selected = {name: float(grid[best[j]]) for j, name in enumerate(METHODS[:2])}
    rows = [{"method": name, "lr0": float(value), "score": float(score[j, i]),
             "full_pilot_mean_gap": float(full_mean[j, i]),
             "selected": int(i == best[j])}
            for j, name in enumerate(METHODS[:2]) for i, value in enumerate(grid)]
    return selected, rows


def simulate(config, learning_rates):
    """Run the four methods with common iid samples at every update."""
    indices = np.unique(np.r_[0, np.geomspace(1, config.steps,
                          min(config.steps, config.checkpoints)).astype(np.int64), config.steps])
    outputs = np.empty((len(indices), len(METHODS), config.runs))
    query_points = np.empty_like(outputs)
    outputs[0] = query_points[0] = config.x0
    x = np.full((2, config.runs), config.x0)
    m, v, vmax = (np.zeros_like(x) for _ in range(3))
    xp = np.full_like(x, config.x0)
    y = np.full_like(x, config.y0)
    p = np.full_like(x, config.p0)
    lr = np.array([learning_rates[name] for name in METHODS[:2]])[:, None]
    ams = np.array([False, True])[:, None]
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 1]))
    next_record = 1
    min_p, max_y = config.p0, abs(config.y0)
    start = time.monotonic()
    for k in range(1, config.steps + 1):
        multiplier = oracle_multiplier(rng, config.runs, config.noise_level)
        old_x = x
        x, m, v, vmax = baseline_step(x, m, v, vmax,
            multiplier * full_gradient(x), k, lr, ams)
        sync = shang_step(xp[0], y[0], p[0], multiplier,
                          config.moment_factor, config.radius, synchronous=True)
        lagged = shang_step(xp[1], y[1], p[1], multiplier,
                            config.moment_factor, config.radius)
        xp = np.stack((sync[0], lagged[0]))
        y = np.stack((sync[1], lagged[1]))
        p = np.stack((sync[2], lagged[2]))
        min_p = min(min_p, float(np.min(p)))
        max_y = max(max_y, float(np.max(np.abs(y))))
        if (not np.all(np.isfinite(xp)) or not np.all(np.isfinite(x))
                or not np.all(np.isfinite(p)) or not np.all(p > 0)
                or not np.all(np.isfinite(y))):
            raise FloatingPointError(f"Invalid state at update {k}; no metric floor was imposed")
        if k == indices[next_record]:
            outputs[next_record, :2] = x
            outputs[next_record, 2:] = xp
            query_points[next_record, :2] = old_x
            query_points[next_record, 2] = sync[3]
            query_points[next_record, 3] = lagged[3]
            next_record += 1
        if k % max(1, config.steps // 10) == 0:
            print(f"  {k:>9,}/{config.steps:,} updates ({time.monotonic()-start:.1f}s)", flush=True)
    diagnostics = {"min_metric": min_p, "max_abs_y": max_y,
                   "projection_bound": config.radius / 2.0}
    return indices, outputs, query_points, diagnostics


def make_plots(config, outdir, indices, outputs):
    """Pointwise medians of gaps and iterates are computed separately."""
    x = np.median(outputs, axis=-1)
    gap = np.median(objective_gap(outputs), axis=-1)
    title = f"Stochastic: median over {config.runs} runs"
    prefix = "reddi_convex_stoch_median"
    for quantity, values, ylabel in (
        ("gap", gap, "Median objective gap"),
        ("iterates", x, "Median output iterate")):
        fig, ax = plt.subplots(figsize=(7.6, 5.3), layout="constrained")
        for j, name in enumerate(METHODS):
            color, linestyle = STYLES[name]
            ax.plot(indices[1:], values[1:, j], color=color, ls=linestyle,
                    lw=1.9, label=name)
        ax.set_xscale("log")
        ax.set_xlim(1, max(2, config.steps))
        if quantity == "gap":
            ax.set_ylim(0.0, 2.0)
            ax.set_title(title)
        else:
            ax.axhline(X_STAR, color="black", ls="--", lw=0.9, label=r"$x^\star=-1$")
            ax.set_ylim(-1.05, 1.05)
            ax.set_title(title)
        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", color="0.8", alpha=0.6, ls=":", lw=0.6)
        ax.legend(loc="best", framealpha=0.9)
        for ext in ("png", "pdf"):
            fig.savefig(outdir / f"{prefix}_{quantity}.{ext}", dpi=200)
        plt.close(fig)


def run_experiment(config, outdir):
    config.validate()
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"Mode=iid stochastic; n={config.noise_level}; L={SMOOTHNESS:g}; "
          f"R={config.radius:g}; C=1+sigma1^2={config.moment_factor:g}; "
          f"P0={config.p0:.8g}", flush=True)
    print("Selecting Adam/AMSGrad base learning rates on separate pilot runs...", flush=True)
    selected, search_rows = tune_baselines(config)
    print(f"Selected: {selected}", flush=True)
    indices, outputs, query_points, diagnostics = simulate(config, selected)
    if np.any(np.abs(outputs) > 1.0) or np.any(np.abs(query_points) > 1.0):
        raise FloatingPointError("Saved output or gradient query lies outside [-1,1]")
    gaps = objective_gap(outputs)
    if not np.all(np.isfinite(gaps)) or np.any(gaps < 0):
        raise FloatingPointError("Invalid saved objective gaps; no figures were generated")
    np.savez_compressed(outdir / "trajectories.npz", iterations=indices,
                        methods=np.array(METHODS), outputs=outputs,
                        query_points=query_points, gaps=gaps)
    with (outdir / "grid_search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=search_rows[0].keys())
        writer.writeheader()
        writer.writerows(search_rows)
    final_rows = [{"method": name, "run": r, "final_x": float(outputs[-1, j, r]),
                   "final_gap": float(gaps[-1, j, r])}
                  for j, name in enumerate(METHODS) for r in range(config.runs)]
    with (outdir / "final_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_rows[0].keys())
        writer.writeheader()
        writer.writerows(final_rows)
    n = config.noise_level
    metadata = {**asdict(config), "stochastic": True,
                "L": SMOOTHNESS, "x_star": X_STAR, "f_star": -1.0,
                "problem": "minimize f(x)=x over [-1,1]: stochastic convex toy",
                "benchmark_reference": "Taniguchi et al., NeurIPS 2024, Section 5",
                "methods": METHODS,
                "sample_slopes": [float(n**2), float(-n)],
                "sample_probabilities": [1.0 / n, 1.0 - 1.0 / n],
                "slope_mean": 1.0, "slope_second_moment": config.moment_factor,
                "smoothness_bound_role": "chosen positive Lipschitz bound for the constant gradient",
                "moment_factor": config.moment_factor,
                "sigma0": 0.0, "sigma1_squared": config.moment_factor - 1.0,
                "P0": config.p0, "selected_lr0": selected,
                "lr_grid": LR_GRID, "betas": [0.9, 0.999], "epsilon": 1e-8,
                "baseline_warmup": False, "weight_decay": 0.0,
                "pilot_criterion": "time mean+std of ensemble mean gap in final 20%",
                "pilot_exact_tie_rule": "lower full-pilot mean gap, then grid index",
                "reported_output": "corrected xplus for SHANG variants; updated x for baselines",
                "saved_data": "all runs at declared logarithmic checkpoints, including initial/final",
                "summary": "pointwise median (objective gaps and iterates separately)",
                "gap_axis": "linear [0,2], logarithmic iteration axis; no plotting floor",
                "primal_constraint": "every method projects its updated output onto [-1,1]",
                "experiment_scope": "stability comparison using theory-derived coupled SHANG parameters",
                "additional_primal_projection": True,
                "unconstrained_theorem_covered": False,
                "synchronous_theorem_covered": False,
                "core_sha256": source_sha256,
                "operation": "simulation",
                **diagnostics}
    with (outdir / "settings.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    make_plots(config, outdir, indices, outputs)
    for j, name in enumerate(METHODS):
        print(f"{name:14s}: median final gap={np.median(gaps[-1,j]):.8e}, "
              f"median x={np.median(outputs[-1,j]):.8g}", flush=True)
    print(f"Saved to {outdir}", flush=True)
    return outdir


def replot_saved_run(source_dir, outdir):
    """Subset existing stochastic data by method name; do not run a simulation."""
    source_dir = Path(source_dir).resolve()
    settings_path = source_dir / "settings.json"
    data_path = source_dir / "trajectories.npz"
    source_settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if source_settings.get("stochastic") is not True:
        raise ValueError("Replotting requires a saved stochastic experiment")
    if source_settings.get("x_star") != X_STAR or source_settings.get("f_star") != -1.0:
        raise ValueError("Source data must use f(x)=x on [-1,1]")
    config = Config(**{key: source_settings[key] for key in Config.__dataclass_fields__
                       if key in source_settings})
    config.validate()
    with np.load(data_path, allow_pickle=False) as saved:
        source_methods = saved["methods"].tolist()
        if any(source_methods.count(name) != 1 for name in METHODS):
            raise ValueError("Each active method must occur exactly once in the source")
        keep = [source_methods.index(name) for name in METHODS]
        indices = saved["iterations"].copy()
        outputs = saved["outputs"][:, keep, :].copy()
        query_points = saved["query_points"][:, keep, :].copy()
        gaps = saved["gaps"][:, keep, :].copy()
    expected_shape = (len(indices), len(METHODS), config.runs)
    if outputs.shape != expected_shape or query_points.shape != expected_shape:
        raise ValueError("Source arrays do not match the recorded run count")
    if indices[0] != 0 or indices[-1] != config.steps or np.any(np.diff(indices) <= 0):
        raise ValueError("Source checkpoints are inconsistent with the recorded horizon")
    if (not np.all(np.isfinite(outputs)) or not np.all(np.isfinite(query_points))
            or np.any(np.abs(outputs) > 1.0) or np.any(np.abs(query_points) > 1.0)
            or not np.array_equal(gaps, objective_gap(outputs))):
        raise ValueError("Source trajectories or objective gaps failed verification")
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(outdir / "trajectories.npz", iterations=indices,
                        methods=np.array(METHODS), outputs=outputs,
                        query_points=query_points, gaps=gaps)
    for filename in ("grid_search.csv", "final_runs.csv"):
        with (source_dir / filename).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = [row for row in reader if row["method"] in METHODS]
        with (outdir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    # Copy only settings belonging to the retained benchmark and methods.
    retained_keys = set(asdict(config)) | {
        "stochastic", "L", "x_star", "f_star", "sample_slopes", "sample_probabilities",
        "slope_mean", "slope_second_moment", "smoothness_bound_role", "moment_factor",
        "sigma0", "sigma1_squared", "P0", "selected_lr0", "lr_grid", "betas", "epsilon",
        "baseline_warmup", "weight_decay", "pilot_criterion", "pilot_exact_tie_rule",
        "reported_output", "saved_data", "summary", "gap_axis", "primal_constraint",
        "experiment_scope", "additional_primal_projection", "unconstrained_theorem_covered",
        "synchronous_theorem_covered", "min_metric", "max_abs_y", "projection_bound"}
    metadata = {key: value for key, value in source_settings.items() if key in retained_keys}
    metadata.update({
        "problem": "minimize f(x)=x over [-1,1]: stochastic convex toy",
        "benchmark_reference": "Taniguchi et al., NeurIPS 2024, Section 5",
        "methods": METHODS,
        "operation": "subset_only_replot",
        "simulation_rerun": False,
        "source_directory": str(source_dir),
        "source_core_sha256": source_settings.get("core_sha256",
                                                   source_settings.get("source_core_sha256")),
        "source_settings_sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest(),
        "source_trajectories_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "plotting_core_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "transformation": "method-axis subset only; retained trajectories and gaps unchanged"})
    with (outdir / "settings.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    make_plots(config, outdir, indices, outputs)
    print(f"Replotted {len(METHODS)} unchanged methods from {source_dir}")
    print(f"Saved to {outdir}")
    return outdir


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=10000000)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--pilot-steps", type=int, default=30000)
    parser.add_argument("--pilot-runs", type=int, default=30)
    parser.add_argument("--radius", type=float, default=2.0,
                        help="projection radius; fixed to 2 for the feasible interval [-1,1]")
    parser.add_argument("--noise-level", type=int, default=50,
                        help="stochastic toy parameter n (positive integer)")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--checkpoints", type=int, default=2500)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replot-from", type=Path,
                        help="subset and replot a saved stochastic run without simulation")
    args = parser.parse_args()
    config = Config(steps=args.steps, runs=args.runs,
                    pilot_steps=args.pilot_steps, pilot_runs=args.pilot_runs,
                    radius=args.radius, noise_level=args.noise_level,
                    seed=args.seed, checkpoints=args.checkpoints)
    outdir = args.output or Path(__file__).resolve().parent / (
        f"reddi_convex_stochastic_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if args.replot_from is not None:
        replot_saved_run(args.replot_from, outdir)
    else:
        run_experiment(config, outdir)


if __name__ == "__main__":
    main()
