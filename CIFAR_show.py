"""
Plot existing CIFAR checkpoints without training or downloading data.

Expected layout:
    {input_dir}/{dataset}-{model}_batch{batch_size}_{optimizer}/seed{seed}/checkpoint_{epoch}.pth

Defaults use CIFAR100-ResNet50, batch 128, seeds 23/24/25, and checkpoint_50.pth.
Legacy directories 0/1/2 are supported when a method has no seed directories.
The new and legacy layouts are never combined for the same method.
Use --ablation to include both beta=0, gamma=1 variants. Missing runs are
reported and skipped; if no valid checkpoint is found, no figures are written.
Only load checkpoints from trusted sources.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


METHODS = ["AdamW", "Adam", "SHANG++", "Adam-SHANG-s", "Adam-SHANG"]
ABLATIONS = ["Adam-SHANG-s(beta=0,gamma=1)", "Adam-SHANG(beta=0,gamma=1)"]
COLOR_MAP = {
    "SHANG++": "purple",
    "AdamW": "blue",
    "Adam": "green",
    "Adam-SHANG-s": "orange",
    "Adam-SHANG": "red",
    "Adam-SHANG-s(beta=0,gamma=1)": "darkgoldenrod",
    "Adam-SHANG(beta=0,gamma=1)": "firebrick",
}


def checkpoint_paths(args, prefix, name):
    folder = Path(args.input_dir) / f"{prefix}{name}"
    filename = f"checkpoint_{args.checkpoint_epoch}.pth"
    if args.runs is not None:
        # Explicit legacy-layout mode for old runs with arbitrary numbering.
        return [folder / str(run) / filename for run in range(args.runs)]
    seed_paths = [folder / f"seed{seed}" / filename for seed in args.seeds]
    if any(path.parent.is_dir() for path in seed_paths) or any(folder.glob("seed*")):
        return seed_paths
    # Original scripts mapped run 0/1/2 to seeds 23/24/25. Preserve that
    # mapping even if a caller selects a subset, e.g. --seeds 24 25.
    legacy_runs = {23: 0, 24: 1, 25: 2}
    if all(seed in legacy_runs for seed in args.seeds):
        legacy_paths = [folder / str(legacy_runs[seed]) / filename for seed in args.seeds]
        if any(path.is_file() for path in legacy_paths):
            print(f"[Info] {name}: using legacy run directories for seeds {args.seeds}.")
            return legacy_paths
    return seed_paths


def load_metrics(args, prefix, names):
    data = {}
    for name in names:
        data[name] = {"Test Loss": [], "Training Loss": [], "Test Accuracy": []}
        for path in checkpoint_paths(args, prefix, name):
            if not path.is_file():
                print(f"[Warning] Missing checkpoint: {path}")
                continue
            print(f"Loading {path}")
            # These checkpoints contain tensors plus ordinary metric lists.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            required = ("train_losses", "test_losses", "test_accuracies")
            if any(key not in checkpoint or len(checkpoint[key]) == 0 for key in required):
                raise ValueError(f"Checkpoint has missing or empty metric histories: {path}")
            data[name]["Test Loss"].append(checkpoint["test_losses"])
            data[name]["Test Accuracy"].append(checkpoint["test_accuracies"])

            # Preserve the existing EMA (including its first-value initialization).
            running_averages = []
            last = checkpoint["train_losses"][0]
            for value in checkpoint["train_losses"]:
                last = args.decay * last + (1 - args.decay) * value
                running_averages.append(last)
            data[name]["Training Loss"].append(running_averages)

        if not data[name]["Test Accuracy"]:
            print(f"[Warning] {name} has no valid runs; skip.")
            continue
        # As before, align each metric by truncating runs to the shortest history.
        for metric, histories in data[name].items():
            length = min(len(history) for history in histories)
            data[name][metric] = np.array([history[:length] for history in histories])
        data[name]["Test Accuracy"] *= 100

    if not any(len(result["Test Accuracy"]) for result in data.values()):
        raise FileNotFoundError(
            f"No usable checkpoint_{args.checkpoint_epoch}.pth found under "
            f"{args.input_dir!r} with prefix {prefix!r}. "
            "Check --input-dir, --dataset, --batch-size, and --checkpoint-epoch. "
            "No plots were written."
        )
    return data


def plot_metrics(data, args, prefix):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "font.size": 16, "axes.labelsize": 18, "axes.titlesize": 16,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 14,
    })
    # Retain the original training-step axis convention and statistical formulas.
    total_steps = (args.train_size / args.batch_size) * args.epochs
    plot_title = prefix.rstrip("_").replace("-", " ").replace("_", " ")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for metric in ("Test Accuracy", "Test Loss", "Training Loss"):
        plt.figure(figsize=(7, 5))
        for name, result in data.items():
            if len(result[metric]) == 0:
                continue
            mean = np.mean(result[metric], axis=0)
            std = np.std(result[metric], axis=0)
            if metric == "Test Accuracy":
                print(f"{name}: final accuracy mean={mean[-1]:g}, std={std[-1]:g}, "
                      f"runs={len(result[metric])}")
            elif metric == "Test Loss":
                mean = np.clip(mean, 1e-8, None)
            x_vals = np.linspace(0, total_steps, len(mean))
            plot = plt.plot if metric == "Test Accuracy" else plt.semilogy
            plot(x_vals, mean, label=name, color=COLOR_MAP[name], linewidth=2.5,
                 linestyle="--" if name in ABLATIONS else "-")
            plt.fill_between(x_vals, mean + std, mean - std,
                             alpha=0.2, color=COLOR_MAP[name])

        plt.xlabel("Training steps")
        plt.ylabel("Test Accuracy (%)" if metric == "Test Accuracy" else "Loss (log scale)")
        plt.title(f"{plot_title} {metric}")
        if metric == "Test Loss":
            plt.legend(loc="upper right", frameon=True)
        elif metric == "Training Loss":
            plt.legend(loc="lower left")
        else:
            plt.legend()
        plt.tight_layout()
        path = Path(args.output_dir) / (prefix + "_" + metric.replace(" ", "_") + ".png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
        if args.show:
            plt.show()
        plt.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot saved CIFAR metric histories; no training or download.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default=None,
                        help="Checkpoint root; defaults to results/cifar100 or results/cifar10.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--dataset", choices=["CIFAR100", "CIFAR10"], default="CIFAR100")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs represented by the plotted checkpoint.")
    parser.add_argument("--checkpoint-epoch", type=int, default=None,
                        help="Checkpoint suffix; defaults to --epochs.")
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--seeds", type=int, nargs="+", default=[23, 24, 25],
                           help="Seeds to load from seedN folders.")
    run_group.add_argument("--runs", type=int, default=None,
                           help="Legacy-only mode: load numbered folders 0 through N-1.")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--decay", type=float, default=0.999,
                        help="EMA coefficient for training loss.")
    parser.add_argument("--prefix", default=None,
                        help="Override the dataset/model/batch checkpoint-directory prefix.")
    parser.add_argument("--ablation", action="store_true",
                        help="Also plot both beta=0, gamma=1 variants.")
    parser.add_argument("--show", action="store_true", help="Display saved plots interactively.")
    args = parser.parse_args(argv)
    if args.input_dir is None:
        args.input_dir = str(Path("results") / args.dataset.lower())
    if args.checkpoint_epoch is None:
        args.checkpoint_epoch = args.epochs
    if min(args.batch_size, args.epochs, args.checkpoint_epoch, args.train_size) < 1:
        parser.error("Batch size, epochs, checkpoint epoch, and train size must be positive.")
    if args.runs is not None and args.runs < 1:
        parser.error("--runs must be positive.")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct nonnegative integers.")
    if not np.isfinite(args.decay) or not 0 <= args.decay < 1:
        parser.error("--decay must be finite and in [0, 1).")
    if args.prefix is not None and (
        not args.prefix or "/" in args.prefix or "\\" in args.prefix
    ):
        parser.error("--prefix must be a nonempty directory-name prefix, not a path.")
    return args


def main(argv=None):
    args = parse_args(argv)
    model = "ResNet50" if args.dataset == "CIFAR100" else "ResNet34"
    prefix = args.prefix or f"{args.dataset}-{model}_batch{args.batch_size}_"
    names = METHODS + ABLATIONS if args.ablation else METHODS
    try:
        data = load_metrics(args, prefix, names)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    plot_metrics(data, args, prefix)


if __name__ == "__main__":
    main()
