# Adam-SHANG: experiment code

Research code for **Adam-SHANG: A Convergent Adam-Type Method for Stochastic
Smooth Convex Optimization**.

The repository contains convex benchmarks, an online stress test, and practical
Transformer/ResNet experiments. The NumPy convex implementations and PyTorch
practical optimizers are separate: the latter are empirical extensions, not
implementations of every hypothesis in the convex theorem.

## Setup

Run commands from the repository root, preferably in a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Smoke tests used Python 3.11, PyTorch 2.7.1, torchvision 0.22.0, NumPy 2.3.2,
Matplotlib 3.10.3, requests 2.32.4, and Pillow 11.3.0. These are validation
versions, not a claim about the environment of every historical paper run.
Install a compatible PyTorch/torchvision pair for your hardware.

Convex and online scripts need only NumPy and Matplotlib. Optional `numba`
acceleration is strongly recommended for the long online experiment; otherwise
the script uses its slower Python fallback. Neural experiments support CPU,
CUDA, and Apple MPS; an accelerator is recommended for full runs.

Importing maintained scripts does not launch training or download data.
Use `--help` for each script's arguments. Default experiments are substantial;
short-budget checks below are functionality tests, not paper results.

## Files

| File | Purpose |
| --- | --- |
| `AdamSHANG.py` | Practical lagged, synchronous, and projected PyTorch optimizers |
| `SHANG_PlusPlus.py` | Non-adaptive SHANG++ comparison |
| `convex_sigma0_zero.py` | Convex benchmark without additive noise |
| `convex_sigma0_nonzero.py` | Convex benchmark with additive noise |
| `reddi_synthetic_stress_test.py` | Online stochastic stress test |
| `transformer_text8.py` | Main Transformer comparison and optional ablations |
| `transformer_text8_projection.py` | Projected/unprojected comparison, batches 32 and 128 |
| `CIFAR100_ResNet50.py`, `CIFAR10_ResNet34.py` | Image-classification experiments |
| `text8_grid_search.py`, `CIFAR_grid_search.py` | Preliminary search utilities |
| `CIFAR_show.py` | Plot saved CIFAR checkpoints |
| `legacy/` | Retired exploratory convex reformulations, not current paper experiments |
| `tests/` | Small, offline CPU checks |

ADOPT is not included: its implementation was absent from the supplied source
folder, and stale references have been removed from maintained experiments.
Datasets, checkpoints, generated figures, and result files are not committed.

## Convex benchmarks

```bash
python convex_sigma0_zero.py --output results/convex_sigma0_zero
python convex_sigma0_nonzero.py --output results/convex_sigma0_nonzero
```

Defaults: dimension 16, 100,000 recorded points including initialization, 200
stochastic trajectories (one when noise-free), and 30,000-point preliminary
stepsize searches. Without additive noise, `sigma1 = 0, 10, 30`; with additive
noise, `sigma1 = 10` and `sigma0 = 0.5, 1, 3`.

Comparisons: SGD, SHANG, Adam, and both Adam-SHANG variants with spectral and
trace-ratio steps. The trace safety factor is computed from the dimension as
`lambda = sqrt(2 / (sqrt(d) + 1))`. Each trajectory uses its own metric. Both
Adam-SHANG variants report objective values at the corrected point `x_plus`.
Final-loss and metric-floor diagnostic CSV files accompany the figures.

Small checks:

```bash
python convex_sigma0_zero.py --steps 40 --pilot-steps 20 --runs 2 --output results/smoke_zero
python convex_sigma0_nonzero.py --steps 40 --pilot-steps 20 --runs 2 --output results/smoke_nonzero
```

The original no-additive-noise script ran dimensions 8 and 16 sequentially. Use
`--dims 8 16` to retain that ordering. Running only dimension 16 changes which
random numbers it receives, since the dimension-8 run no longer consumes them.

## Online stress test

```bash
python reddi_synthetic_stress_test.py --output results/reddi_online
python reddi_synthetic_stress_test.py --steps 1000 --runs 2 --checkpoints 100 --output results/smoke_online
```

The default is 10,000,000 rounds, 30 runs, and 10,000 recording checkpoints.
Adam, AMSGrad, Adam-SHANG, and Adam-SHANG-s are compared. Base scales are constant.
Optional pilot-search switches in `SETTINGS` are off; `--pilot-steps` and
`--pilot-runs` change budgets but do not enable searches.

Saved **hindsight** average regret is
`(sum(g_s * x_s) + abs(sum(g_s))) / t` for the interval `[-1, 1]`. This differs
from finite-sample regret against a fixed comparator `x = -1`. The example is
an online stress test outside the fixed-objective convex theory.

## text8

Main scripts read `data/text8.txt`. Supply an existing file with `--data-path`
or permit downloading with `--download`:

```bash
python transformer_text8.py --download --batch-size 32
python transformer_text8.py --batch-size 128
python transformer_text8_projection.py --batch-sizes 32 128
```

The main model is a four-layer Pre-LayerNorm Transformer, hidden size 256,
eight heads, feedforward size 1024, sequence length 256. Defaults are 30,000
steps and seeds 23, 24, 25. The chronological split is 90/10; every 2,000 steps,
validation uses the first 100 batches.

The main comparison includes both SHANG variants, Adam, AdamW, and scheduled
Adam/AdamW (`Adam_lr`, `AdamW_lr`). The SHANG variants use no external schedule
in this final experiment. `--ablation` adds their `beta=0, gamma=1` variants.

The projection script runs both SHANG variants with and without projection,
with `R=10` and batches 32 and 128. Use `--projection-r` to change the radius.
Each invocation creates a timestamped directory under `results/text8/` or
`results/text8_projection/` with configuration, CSV results, and figures.

Use `--steps`, `--seeds`, `--eval-every`, `--eval-batches`, `--optimizers`, and
`--device cpu` to reduce a run. It still needs a sufficiently long corpus.

## CIFAR

```bash
python CIFAR100_ResNet50.py --download --batch-size 128
python CIFAR100_ResNet50.py --batch-size 32
python CIFAR10_ResNet34.py --download --batch-size 128
python CIFAR10_ResNet34.py --batch-size 32
```

Data default to `data/cifar`; use `--data-dir` for an existing location. Downloads
require `--download`. Defaults are 50 epochs and seeds 23, 24, 25. The five
methods are both SHANG variants, Adam, AdamW, and SHANG++; `--ablation` adds the
two `beta=0, gamma=1` variants. No external learning-rate schedules are used.

Example saved checkpoint:

```text
results/cifar100/CIFAR100-ResNet50_batch128_Adam-SHANG/seed23/checkpoint_50.pth
```

Plot completed runs:

```bash
python CIFAR_show.py --input-dir results/cifar100 --output-dir results/cifar100/plots --dataset CIFAR100 --batch-size 128
python CIFAR_show.py --input-dir results/cifar10 --output-dir results/cifar10/plots --dataset CIFAR10 --batch-size 128
```

Match `--epochs`, `--checkpoint-epoch`, and batch size to the run. Add
`--ablation` for those experiments. Load only trusted checkpoints.

## Preliminary hyperparameter searches

Inspect workloads without loading data or training:

```bash
python text8_grid_search.py --dry-run
python CIFAR_grid_search.py --dry-run
```

The text8 utility runs a **full Cartesian grid**, not an automated staged search:

- `lr`: `1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001`;
- SHANG `beta`: `0.1, 0.5, 0.01, 0.05, 0.001, 0.005`;
- SHANG `gamma`: `0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 1`.

There are 294 candidates per SHANG variant and seven each for Adam and AdamW
(602 total). It uses the first five million characters, a chronological 90/10
split, a two-layer width-128 Transformer, 2,000 steps, and final validation
cross-entropy over the first 30 batches. All candidates, including SHANG, use
warmup/cosine **in this preliminary utility**. Selection minimizes that final
validation cross-entropy.

The CIFAR utility defaults to Adam-SHANG-s on CIFAR-100/ResNet-50, fixes `lr=0.1`,
and searches four `beta` values (`0.05, 0.01, 0.5, 0.1`) and seven `gamma` values
(`0.005, 0.001, 0.05, 0.01, 0.5, 0.1, 1`): 28 candidates. It runs 20 epochs on
a 20,000-example training subset, batch 64, recording **training loss** for
comparison. It does not select using held-out test accuracy.

```bash
python text8_grid_search.py --data-path data/text8.txt --output-dir results/text8_grid
python CIFAR_grid_search.py --data-dir data/cifar --output-dir results/cifar_grid
```

Add `--download` if data are absent. `--max-configs 1` and shorter
`--steps`/`--epochs` are useful checks. These utilities are not a complete
automated record of historical staged/manual searches for every method. Final
settings are encoded in the main scripts; search outputs are not auto-loaded.

## Reproduction notes

- Cleanup preserves the original optimizer update formulas. Small-budget
  regression checks against the pre-cleanup scripts produced identical convex
  CSV values and identical online trajectory arrays.
- The convex Adam baseline initializes its second-moment accumulator to `eps`
  (`1e-8`), not exactly zero. That existing implementation is retained; distinguish
  it from a denominator-only epsilon when describing the experiment.
- Online outputs use the hindsight comparator above, not a fixed comparator.
- Practical algorithms, exploratory search utilities, and historical search
  descriptions are not interchangeable. Check the script and configuration
  before claiming reproduction of a particular table.
- Smoke tests validate execution, not scientific conclusions. Full paper-scale
  experiments were not rerun during cleanup.
- Generated outputs are ignored by Git. Keep separate copies of results needed
  for archival reproducibility.

## Attribution

`util.py` retains its original attribution to Matthias Wright's
[`cifar10-resnet`](https://github.com/matthias-wright/cifar10-resnet) project.
See `THIRD_PARTY_NOTICES.md`. Dataset and dependency terms apply independently.
This cleanup does not assign a new blanket license to the research code.
