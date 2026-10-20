"""
Text8 grid search for Adam-SHANG experiments.

This script performs a lightweight hyperparameter sweep on a reduced text8
Transformer language-modeling task. The purpose is to select hyperparameters
for the full text8 experiments, not to report final benchmark numbers.

Compared optimizers:
    - Adam-SHANG
    - Adam-SHANG-s
    - Adam
    - AdamW

Learning-rate protocol:
    - Every candidate receives the same training and validation budget.
    - Adam-SHANG, Adam-SHANG-s, Adam, and AdamW use linear warmup followed by
      cosine decay during this lightweight sweep.
    - The lr values in SEARCH_SPACE are base/peak learning rates.
    - The grid search does not distinguish Adam/AdamW with vs. without a
      schedule; that comparison belongs to the full experiment script.

Dataset:
    - text8, optionally downloaded from https://mattmahoney.net/dc/text8.zip
      with --download; downloads are disabled by default.
    - A prefix of the full text8 corpus is used for the sweep.
    - The first 90% of tokens are used for training and the last 10% for
      validation.

Reproducibility:
    - The sweep seed is specified in TRAIN_CFG["seed"].
    - Each hyperparameter combination is evaluated with the same model seed and
      the same DataLoader shuffle seed for a fair quick comparison.
    - Results are saved to CSV, and one plot is generated for each optimizer.
    - The scheduler is applied uniformly to all optimizers during the sweep.
    - The current script searches the full Cartesian grid, not an automatic
      staged search. Use --dry-run to inspect its cost before training.

"""

import argparse
import csv
import itertools
import os
import re
import shutil
import tempfile
import time
import zipfile
from typing import Dict, Iterable, List, Tuple
import torch
import math
import requests
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from AdamSHANG import AdamSHANG, AdamSHANGs

# Device selection

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Set by main; importing this module does not initialize an accelerator.
DEVICE = torch.device("cpu")

# Output and data paths
OUTPUT_DIR = "text8_grid_search_outputs"
TEXT8_PATH = os.path.join("data", "text8.txt")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "text8_grid_search_results.csv")
BEST_CSV = os.path.join(OUTPUT_DIR, "text8_grid_search_best_summary.csv")

# Search spaces.
# Put tuned parameters in "grid" and fixed parameters in "fixed".
# All methods use the sweep's linear-warmup + cosine-decay schedule.
COMMON_BASE_STEP_GRID = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

SEARCH_SPACE = {
    "AdamSHANG": {
        "grid": {
            "lr": COMMON_BASE_STEP_GRID,
            "beta":    [0.1, 0.5, 0.01, 0.05, 0.001, 0.005],
            "gamma": [0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 1],
        },
        "fixed": {
            "weight_decay": 1e-2
        },
    },
    "AdamSHANGs": {
        "grid": {
            "lr": COMMON_BASE_STEP_GRID,
            "beta":    [0.1, 0.5, 0.01, 0.05, 0.001, 0.005],
            "gamma": [0.1, 0.5, 0.01, 0.05, 0.001, 0.005, 1],
        },
        "fixed": {
            "weight_decay": 1e-2
        },
    },
    "Adam": {
        "grid": {
            # The common grid includes deliberately unstable large candidates.
            "lr": COMMON_BASE_STEP_GRID,
        },
        "fixed": {
            "betas": (0.9, 0.999),
            "weight_decay": 1e-5,
        },
    },
    "AdamW": {
        "grid": {
            "lr": COMMON_BASE_STEP_GRID,
        },
        "fixed": {
            "betas": (0.9, 0.999),
            "weight_decay": 1e-2,
        },
    },
}


# Lightweight training configuration for the sweep.
TRAIN_CFG = dict(
    seq_len=128,
    data_limit=5_000_000,
    d_model=128,
    n_heads=4,
    n_layers=2,
    ffn_dim=512,
    dropout=0.1,
    batch_size=128,
    sweep_steps=2000,
    warmup_steps=200,
    eval_batches=30,
    grad_clip=1.0,
    seed=23,
)


# Data
def download_text8(path: str = TEXT8_PATH):
    if os.path.exists(path):
        return

    data_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(data_dir, exist_ok=True)
    print("Downloading text8 (~30MB)...")
    with tempfile.TemporaryDirectory(prefix=".text8-download-", dir=data_dir) as tmp:
        zip_path = os.path.join(tmp, "text8.zip")
        with requests.get("https://mattmahoney.net/dc/text8.zip", stream=True,
                          timeout=(10, 120)) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as handle:
                for chunk in response.iter_content(65536):
                    handle.write(chunk)
        extracted_path = os.path.join(tmp, "text8.txt")
        # Read only the expected corpus member; never extract arbitrary paths.
        with zipfile.ZipFile(zip_path) as archive, archive.open("text8") as source:
            with open(extracted_path, "wb") as target:
                shutil.copyfileobj(source, target)
        os.replace(extracted_path, path)
    print(f"text8 saved to {path}")


class Text8Dataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, i):
        return (
            self.data[i : i + self.seq_len],
            self.data[i + 1 : i + self.seq_len + 1],
        )


_DATA_CACHE = {}

def prepare_data(cfg, path=TEXT8_PATH, download=False):
    """Load and encode text8 once, downloading only when explicitly allowed."""
    cache_key = (os.path.abspath(path), cfg["data_limit"])
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    if not os.path.isfile(path):
        if not download:
            raise FileNotFoundError(
                f"{path} not found. Supply --data-path or allow --download."
            )
        download_text8(path)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()[: cfg["data_limit"]]

    vocab = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    encoded = torch.tensor([char2idx[c] for c in text], dtype=torch.long)

    split = int(len(encoded) * 0.9)
    train_tok = encoded[:split]
    val_tok = encoded[split:]

    print(f"Total chars used: {len(text):,}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Train tokens: {len(train_tok):,} | Val tokens: {len(val_tok):,}")

    _DATA_CACHE[cache_key] = (train_tok, val_tok, len(vocab))
    return _DATA_CACHE[cache_key]


def make_loaders(train_tok, val_tok, cfg, seed: int):
    """Build DataLoaders with a fixed shuffle generator for fair comparisons."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = torch.cuda.is_available()
    num_workers = 2 if torch.cuda.is_available() else 0

    train_loader = DataLoader(
        Text8Dataset(train_tok, cfg["seq_len"]),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        Text8Dataset(val_tok, cfg["seq_len"]),
        batch_size=cfg["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# Model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class SmallTransformerLM(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        d_model = cfg["d_model"]
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, cfg["seq_len"] + 10, cfg["dropout"])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=cfg["n_heads"],
                dim_feedforward=cfg["ffn_dim"],
                dropout=cfg["dropout"],
                batch_first=True,
                norm_first=True,
            ),
            num_layers=cfg["n_layers"],
        )
        self.head = nn.Linear(d_model, vocab_size)
        self._scale = math.sqrt(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        h = self.pos(self.embed(x) * self._scale)
        return self.head(self.encoder(h, mask=mask, is_causal=True))


# Optimizer and scheduler utilities

def _clean_optimizer_hparams(hparams: Dict):
    """Remove keys used by the sweep code but not accepted by optimizers."""
    return {
        k: v
        for k, v in hparams.items()
        if k not in {"warmup_steps"}
    }


def make_optimizer(name: str, params: Iterable[torch.nn.Parameter], hparams: Dict):
    opt_hparams = _clean_optimizer_hparams(hparams)

    if name == "AdamSHANG":
        return AdamSHANG(params, **opt_hparams)
    if name == "AdamSHANGs":
        return AdamSHANGs(params, **opt_hparams)
    if name == "Adam":
        return torch.optim.Adam(params, **opt_hparams)
    if name == "AdamW":
        return torch.optim.AdamW(params, **opt_hparams)
    raise ValueError(f"Unknown optimizer: {name}")


def make_scheduler(opt_name, optimizer, cfg):
    """Build the same warmup/cosine sweep schedule for each optimizer."""
    warmup_steps = int(cfg["warmup_steps"])
    total_steps = int(cfg["sweep_steps"])

    def lr_lambda(step):
        # LambdaLR calls this with step starting from 0.
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, criterion, max_batches):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total += criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n += 1
    model.train()
    avg = total / max(n, 1)
    ppl = math.exp(min(avg, 709.0)) if avg < 709.0 else float("inf")
    return avg, ppl


def quick_train(opt_name, hparams, train_tok, val_tok, vocab_size, cfg):
    """Run one lightweight sweep trial and return the validation metrics."""
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    train_loader, val_loader = make_loaders(train_tok, val_tok, cfg, seed=seed)
    model = SmallTransformerLM(vocab_size, cfg).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    try:
        opt = make_optimizer(opt_name, model.parameters(), hparams)
        scheduler = make_scheduler(opt_name, opt, cfg)
    except Exception as e:
        return float("inf"), float("inf"), f"opt_init_error: {e}"

    loader_iter = iter(train_loader)
    status = "ok"

    for step in range(1, cfg["sweep_steps"] + 1):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

        if not math.isfinite(float(loss.item())):
            status = f"diverged@{step}"
            return float("inf"), float("inf"), status

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

        try:
            opt.step()
            scheduler.step()
        except RuntimeError as e:
            status = f"runtime_err@{step}: {e}"
            return float("inf"), float("inf"), status

    val_loss, val_ppl = evaluate(model, val_loader, criterion, cfg["eval_batches"])
    return val_loss, val_ppl, status


# Grid-search loop
def expand_grid(space: Dict) -> Tuple[List[str], List[Tuple], Dict]:
    grid_keys = list(space["grid"].keys())
    grid_values = list(space["grid"].values())
    fixed = dict(space.get("fixed", {}))
    combos = list(itertools.product(*grid_values))
    return grid_keys, combos, fixed


def run_grid_search(opt_name, space, train_tok, val_tok, vocab_size, cfg,
                    max_configs=None):
    grid_keys, combos, fixed = expand_grid(space)
    if max_configs is not None:
        combos = combos[:max_configs]
    total = len(combos)

    print(f"\n{'═' * 72}")
    print(f"  {opt_name} — {total} combinations")
    print(f"{'═' * 72}")

    records = []
    best_val, best_combo = float("inf"), None

    for i, combo in enumerate(combos, 1):
        hparams = dict(zip(grid_keys, combo))
        hparams.update(fixed)
        combo_dict = dict(zip(grid_keys, combo))
        combo_str = "  ".join(f"{k}={v}" for k, v in combo_dict.items())
        t0 = time.time()

        val_loss, val_ppl, status = quick_train(
            opt_name, hparams, train_tok, val_tok, vocab_size, cfg
        )

        elapsed = time.time() - t0
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_combo = dict(combo_dict)

        flag = "*" if improved else " "
        print(
            f"  [{i:3d}/{total}] {flag} {combo_str:<58s} "
            f"val_loss={val_loss:.4f}  ppl={val_ppl:.1f}  "
            f"[{elapsed:.0f}s]  {status}"
        )

        records.append(
            {
                "optimizer": opt_name,
                "schedule": "linear_warmup_cosine",
                **combo_dict,
                **{f"fixed_{k}": v for k, v in fixed.items()},
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "elapsed_sec": elapsed,
                "status": status,
            }
        )

    print(f"\n  Best: {best_combo} -> val_loss={best_val:.4f}")
    return records, best_combo, best_val


# Saving and plotting
def save_csv(records, path):
    if not records:
        return
    fieldnames = ["optimizer"]
    extra_keys = sorted({k for r in records for k in r.keys()} - {"optimizer"})
    fieldnames.extend(extra_keys)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved -> {path}")


def slugify(name: str) -> str:
    name = name.lower().replace("++", "plusplus")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def combo_label(record, grid_keys):
    return ", ".join(f"{k}={record[k]}" for k in grid_keys if k in record)


def plot_grid(records, opt_name, grid_keys, cfg, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not found; skipping plots")
        return

    finite = [r for r in records if math.isfinite(float(r["val_loss"]))]
    if not finite:
        print(f"[{opt_name}] all runs diverged; skipping plot")
        return

    if len(grid_keys) == 1:
        key = grid_keys[0]
        finite = sorted(finite, key=lambda r: r[key])
        xs = [str(r[key]) for r in finite]
        ys = [float(r["val_loss"]) for r in finite]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(xs, ys, alpha=0.75)
        ax.set_xlabel(key)
        ax.set_ylabel("validation loss")
        ax.set_title(f"{opt_name}: validation loss ({cfg['sweep_steps']} steps)")
        ax.tick_params(axis="x", rotation=30)

    elif len(grid_keys) == 2:
        kx, ky = grid_keys
        xs = sorted(set(r[kx] for r in finite))
        ys = sorted(set(r[ky] for r in finite))
        matrix = np.full((len(ys), len(xs)), np.nan)
        lookup = {(r[kx], r[ky]): float(r["val_loss"]) for r in finite}

        for yi, yv in enumerate(ys):
            for xi, xv in enumerate(xs):
                if (xv, yv) in lookup:
                    matrix[yi, xi] = lookup[(xv, yv)]

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        im = ax.imshow(matrix, aspect="auto")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys)
        ax.set_xlabel(kx)
        ax.set_ylabel(ky)
        ax.set_title(f"{opt_name}: validation loss ({cfg['sweep_steps']} steps)")
        fig.colorbar(im, ax=ax, label="validation loss")

        for yi in range(len(ys)):
            for xi in range(len(xs)):
                value = matrix[yi, xi]
                if not math.isnan(value):
                    ax.text(xi, yi, f"{value:.3f}", ha="center", va="center", fontsize=8)

    else:
        # For 3D+ grids, show the best few configurations rather than a misleading
        # heatmap over only two dimensions.
        top = sorted(finite, key=lambda r: float(r["val_loss"]))[:20]
        labels = [combo_label(r, grid_keys) for r in top]
        values = [float(r["val_loss"]) for r in top]

        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(top))))
        y_pos = np.arange(len(top))
        ax.barh(y_pos, values, alpha=0.75)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("validation loss")
        ax.set_title(f"{opt_name}: top configurations ({cfg['sweep_steps']} steps)")

    plt.tight_layout()
    fname = os.path.join(output_dir, f"grid_{slugify(opt_name)}.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {fname}")


def save_best_summary(best_summary, path=BEST_CSV):
    rows = []
    for opt_name, (combo, val) in best_summary.items():
        row = {"optimizer": opt_name, "val_loss": val}
        if combo:
            row.update(combo)
        rows.append(row)
    save_csv(rows, path)


# Main entry point
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Full Cartesian text8 hyperparameter sweep (not staged).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", default=TEXT8_PATH,
                        help="Local text8 text file.")
    parser.add_argument("--download", action="store_true",
                        help="Allow downloading text8 if --data-path does not exist.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--optimizers", nargs="+", choices=list(SEARCH_SPACE),
                        default=list(SEARCH_SPACE))
    parser.add_argument("--steps", type=int, default=TRAIN_CFG["sweep_steps"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CFG["batch_size"])
    parser.add_argument("--data-limit", type=int, default=TRAIN_CFG["data_limit"])
    parser.add_argument("--eval-batches", type=int, default=TRAIN_CFG["eval_batches"])
    parser.add_argument("--warmup-steps", type=int, default=TRAIN_CFG["warmup_steps"])
    parser.add_argument("--seed", type=int, default=TRAIN_CFG["seed"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    parser.add_argument("--max-configs", type=int, default=None,
                        help="Limit each optimizer to its first N grid entries; not a full search.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print search budgets without downloading or training.")
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.data_limit, args.eval_batches) < 1:
        parser.error("Steps, batch size, data limit, and eval batches must be positive.")
    if args.warmup_steps < 0 or args.seed < 0:
        parser.error("Warmup steps and seed must be nonnegative.")
    if args.max_configs is not None and args.max_configs < 1:
        parser.error("--max-configs must be positive.")
    if len(set(args.optimizers)) != len(args.optimizers):
        parser.error("Do not repeat optimizer names.")
    return args


def main(argv=None):
    global DEVICE
    args = parse_args(argv)
    cfg = dict(TRAIN_CFG, sweep_steps=args.steps, batch_size=args.batch_size,
               data_limit=args.data_limit, eval_batches=args.eval_batches,
               warmup_steps=args.warmup_steps, seed=args.seed)
    for name in args.optimizers:
        full_count = len(expand_grid(SEARCH_SPACE[name])[1])
        count = min(args.max_configs, full_count) if args.max_configs else full_count
        print(f"{name}: {count}/{full_count} grid entries x {cfg['sweep_steps']} steps")
    if args.max_configs:
        print("Limited-grid run: this does not reproduce the full hyperparameter search.")
    if args.dry_run:
        return

    DEVICE = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Device: {DEVICE}")
    print("Loading data...")
    train_tok, val_tok, vocab_size = prepare_data(cfg, args.data_path, args.download)
    train_loader, val_loader = make_loaders(
        train_tok, val_tok, cfg, seed=int(cfg["seed"])
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Corpus prefix is too short for a full training and validation batch.")
    print(
        f"vocab={vocab_size} | train_batches={len(train_loader)} | "
        f"val_batches={len(val_loader)}"
    )
    print(
        f"Sweep model: {cfg['n_layers']} layers, d_model={cfg['d_model']}, "
        f"batch={cfg['batch_size']}, steps={cfg['sweep_steps']}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    all_records = []
    best_summary = {}

    for opt_name in args.optimizers:
        space = SEARCH_SPACE[opt_name]
        records, best_combo, best_val = run_grid_search(
            opt_name, space, train_tok, val_tok, vocab_size, cfg, args.max_configs
        )
        all_records.extend(records)
        best_summary[opt_name] = (best_combo, best_val)
        plot_grid(records, opt_name, list(space["grid"].keys()), cfg, args.output_dir)

    print(f"\n{'═' * 72}")
    print("  Best hyperparameters per optimizer")
    print(f"{'═' * 72}")
    for name, (combo, val) in best_summary.items():
        print(f"  {name:<32s} val_loss={val:.4f}  params={combo}")

    save_csv(all_records, os.path.join(args.output_dir, "text8_grid_search_results.csv"))
    save_best_summary(best_summary,
                      os.path.join(args.output_dir, "text8_grid_search_best_summary.csv"))

    print(f"\n{'─' * 72}")
    print("  Copy these values into the full text8 experiment CFG if needed:")
    print(f"{'─' * 72}")
    for name, (combo, _) in best_summary.items():
        if combo:
            print(f"# {name}")
            for k, v in combo.items():
                print(f"#   {k} = {v}")


if __name__ == "__main__":
    main()
