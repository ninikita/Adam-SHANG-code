"""
Text8 character-level language modeling experiment.

This script compares Adam-SHANG and Adam-SHANG-s with Adam, AdamW,
and Adam/AdamW baselines equipped with an external warmup+cosine learning-rate
schedule on the full text8 character-level language modeling benchmark.

Dataset:
    - text8, downloaded from https://mattmahoney.net/dc/text8.zip with --download
    - The full text8 corpus is used.
    - The first 90% of tokens are used for training and the last 10% for
      validation.
    - No additional preprocessing is used beyond character-level encoding.

Model:
    - 4-layer Transformer encoder language model
    - d_model = 256
    - n_heads = 8
    - feedforward dimension = 1024
    - dropout = 0.1
    - approximately 3M trainable parameters

Training protocol:
    - batch size = 32/128
    - sequence length = 256
    - total training steps = 30000
    - gradient clipping threshold = 1.0
    - validation is performed every 2000 steps
    - each validation uses 100 batches
    - Adam_lr and AdamW_lr use an external linear-warmup + cosine-decay
      learning-rate schedule; the other optimizers use their own stepsizes

Reproducibility:
    - The random seeds are specified in CFG["seeds"].
    - Results are written to a CSV file containing optimizer, seed, step,
      training loss, validation loss, and validation perplexity.
    - The script reports mean and standard deviation across seeds at the end.
    - The exact optimizer hyperparameters are specified in CFG.

Compute:
    - The script supports CUDA, Apple MPS, and CPU.
    - Run with --help for training, data, output, and seed options.
    - Add --ablation to include the two beta=0,gamma=1 variants.
"""

import argparse
import os, time, csv, json, requests, zipfile, shutil, tempfile
from collections import defaultdict
from datetime import datetime
import torch
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from AdamSHANG import AdamSHANG, AdamSHANGs

# Device selection
def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# Experiment configuration
CFG = dict(
    seq_len       = 256,
    d_model       = 256,
    n_heads       = 8,
    n_layers      = 4,
    ffn_dim       = 1024,
    dropout       = 0.1,

    batch_size    = 32, # or 128
    total_steps   = 30000,
    warmup_steps  = 1000,
    log_every     = 500,
    eval_every    = 2000,
    eval_batches  = 100,
    grad_clip     = 1.0,

    lr = {
        "AdamSHANG":          0.5,
        "AdamSHANGs":         0.5,
        "AdamSHANG(beta=0,gamma=1)": 0.5,
        "AdamSHANGs(beta=0,gamma=1)":0.5,
        "AdamW":              5e-3,
        "Adam":               5e-3,
        "AdamW_lr":           5e-3,
        "Adam_lr":            5e-3,
    },

    adamshang  = dict(beta=0.05,  gamma=0.001, weight_decay=1e-2),
    adamshangs = dict(beta=0.05,  gamma=0.01,   weight_decay=1e-2),
    adamshang2 = dict(beta=0, gamma=1,   weight_decay=1e-2),
    adamshangs2= dict(beta=0,     gamma=1,     weight_decay=1e-2),
    adamw      = dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2),
    adam       = dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5),
    seeds      = [23,24,25],
    data_path  = "data/text8.txt",
    download   = False,
    output_dir = "results/text8",
)


# Dataset
def download_text8(path="text8.txt"):
    if os.path.exists(path):
        print(f"{path} already exists, skipping download.")
        return
    print("Downloading text8 (~30MB)...")
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".text8-download-", dir=parent_dir) as tmp:
        zip_path = os.path.join(tmp, "text8.zip")
        with requests.get("https://mattmahoney.net/dc/text8.zip", stream=True,
                          timeout=(10, 120)) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as handle:
                for chunk in response.iter_content(65536):
                    handle.write(chunk)
        extracted_path = os.path.join(tmp, "text8.txt")
        with zipfile.ZipFile(zip_path) as archive, archive.open("text8") as source:
            with open(extracted_path, "wb") as target:
                shutil.copyfileobj(source, target)
        os.replace(extracted_path, path)
    print("Download complete.")

class Text8Dataset(Dataset):
    def __init__(self, data, seq_len):
        self.data    = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len - 1
    def __getitem__(self, i):
        return self.data[i : i + self.seq_len], \
               self.data[i + 1 : i + self.seq_len + 1]

_DATA_CACHE = None
_DATA_CACHE_KEY = None

def build_data(cfg):
    global _DATA_CACHE, _DATA_CACHE_KEY
    data_path = cfg.get("data_path", "data/text8.txt")
    cache_key = (os.path.abspath(data_path), cfg["seq_len"], cfg["batch_size"])
    if _DATA_CACHE is not None and _DATA_CACHE_KEY == cache_key:
        return _DATA_CACHE

    if not os.path.isfile(data_path):
        if not cfg.get("download", False):
            raise FileNotFoundError(f"{data_path} not found. Supply --data-path or allow --download.")
        download_text8(data_path)

    print("Reading full text8 (~100M chars)...")
    with open(data_path, encoding="utf-8") as handle:
        text = handle.read()
    print(f"  Total chars: {len(text):,}")

    vocab    = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"  Vocab size: {vocab_size}")

    encoded = torch.tensor([char2idx[c] for c in text], dtype=torch.long)

    split = int(len(encoded) * 0.9)
    train_tok = encoded[:split]
    val_tok   = encoded[split:]
    print(f"  Train tokens: {len(train_tok):,}  |  Val tokens: {len(val_tok):,}")

    pin = torch.cuda.is_available()
    nw  = 2 if torch.cuda.is_available() else 0
    kw  = dict(batch_size=cfg["batch_size"], drop_last=True,
               num_workers=nw, pin_memory=pin)

    train_dl = DataLoader(Text8Dataset(train_tok, cfg["seq_len"]),
                          shuffle=True,  **kw)
    val_dl   = DataLoader(Text8Dataset(val_tok,   cfg["seq_len"]),
                          shuffle=False, **kw)

    print(f"  Train batches: {len(train_dl):,}  |  Val batches: {len(val_dl):,}")
    _DATA_CACHE = (train_dl, val_dl, vocab_size)
    _DATA_CACHE_KEY = cache_key
    return _DATA_CACHE

# Model definition
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])

class SmallTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers,
                 ffn_dim, dropout, seq_len):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos     = PositionalEncoding(d_model, seq_len + 10, dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=ffn_dim, dropout=dropout,
                batch_first=True, norm_first=True),
            num_layers=n_layers)
        self.head   = nn.Linear(d_model, vocab_size)
        self._scale = math.sqrt(d_model)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  Model params: {n_params/1e6:.2f}M")

    def forward(self, x):
        T    = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h    = self.pos(self.embed(x) * self._scale)
        return self.head(self.encoder(h, mask=mask, is_causal=True))

# Helper functions
def build_optimizer(name, params, cfg):
    lr = cfg["lr"][name]
    if name == "AdamSHANG":             return AdamSHANG(params,  lr=lr, **cfg["adamshang"])
    if name == "AdamSHANG(beta=0,gamma=1)":   return AdamSHANG(params,  lr=lr, **cfg["adamshang2"])
    if name == "AdamSHANGs":            return AdamSHANGs(params, lr=lr, **cfg["adamshangs"])
    if name == "AdamSHANGs(beta=0,gamma=1)":  return AdamSHANGs(params, lr=lr, **cfg["adamshangs2"])
    if name in {"AdamW", "AdamW_lr"}:   return torch.optim.AdamW(params, lr=lr, **cfg["adamw"])
    if name in {"Adam", "Adam_lr"}:     return torch.optim.Adam(params,  lr=lr, **cfg["adam"])
    raise ValueError(f"Unknown optimizer: {name}")


SCHEDULED_OPTIMIZERS = {"Adam_lr", "AdamW_lr"}


def build_scheduler(name, optimizer, cfg):
    """
    Build the optional external learning-rate schedule.

    Only Adam_lr and AdamW_lr use this scheduler. The schedule is linear warmup
    for cfg["warmup_steps"] steps followed by cosine decay to zero. This keeps
    the scheduled Adam/AdamW baselines separate from the plain Adam/AdamW
    baselines used without an external schedule.
    """
    if name not in SCHEDULED_OPTIMIZERS:
        return None

    warmup_steps = int(cfg["warmup_steps"])
    total_steps = int(cfg["total_steps"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, criterion, max_batches):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches: break
        x, y   = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total += criterion(logits.reshape(-1, logits.size(-1)),
                           y.reshape(-1)).item()
        n += 1
    model.train()
    avg = total / max(n, 1)
    ppl = math.exp(min(avg, 709.0)) if avg < 709.0 else float("inf")
    return avg, ppl

# Single training run
def run(opt_name, train_loader, val_loader, vocab_size, cfg, seed=23):
    print(f"\n{'─'*60}")
    print(f"  {opt_name}   lr={cfg['lr'][opt_name]}  seed={seed}")
    print(f"{'─'*60}")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = SmallTransformerLM(
        vocab_size, cfg["d_model"], cfg["n_heads"],
        cfg["n_layers"], cfg["ffn_dim"],
        cfg["dropout"], cfg["seq_len"]
    ).to(DEVICE)

    criterion   = nn.CrossEntropyLoss()
    opt         = build_optimizer(opt_name, model.parameters(), cfg)
    scheduler   = build_scheduler(opt_name, opt, cfg)

    history     = dict(step=[], train_loss=[], val_loss=[], val_ppl=[], step_times=[])
    loader_iter = iter(train_loader)
    t0          = time.time()

    for step in range(1, cfg["total_steps"] + 1):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        opt.zero_grad()
        loss = criterion(model(x).reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])

        t_step = time.perf_counter()
        try:
            opt.step()
            if scheduler is not None:
                scheduler.step()
        except RuntimeError as e:
            print(f"  [!] step {step}: {e} — skipping")
            opt.zero_grad()
            continue
        history["step_times"].append(time.perf_counter() - t_step)

        if step % cfg["log_every"] == 0:
            elapsed    = time.time() - t0
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else opt.param_groups[0]["lr"]
            eta_min    = (elapsed / step) * (cfg["total_steps"] - step) / 60
            print(f"  step {step:6d}/{cfg['total_steps']} | "
                  f"loss {loss.item():.4f} | "
                  f"lr {current_lr:.2e} | "
                  f"{elapsed/60:.1f}min  ETA {eta_min:.0f}min")

        if step % cfg["eval_every"] == 0 or step == cfg["total_steps"]:
            vl, vp = evaluate(model, val_loader, criterion, cfg["eval_batches"])
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(vl)
            history["val_ppl"].append(vp)
            avg_ms = 1000 * sum(history["step_times"]) / max(len(history["step_times"]), 1)
            print(f"  ▶ val_loss {vl:.4f}  ppl {vp:.2f}  avg_step {avg_ms:.1f}ms")

    return history

# Plotting utilities
def normalize_name(name: str) -> str:
    """Normalize optimizer names that may appear in older CSV files."""
    name = name.strip()
    alias = {
        "AdamSHANG(b=0,r=1)": "AdamSHANG(beta=0,gamma=1)",
        "AdamSHANGs(b=0,r=1)": "AdamSHANGs(beta=0,gamma=1)",
        "AdamSHANG": "AdamSHANG",
        "AdamSHANGs": "AdamSHANGs",
        "AdamW": "AdamW",
        "Adam": "Adam",
        "AdamW_lr": "AdamW_lr",
        "Adam_lr": "Adam_lr",
    }
    return alias.get(name, name)


def plot_results_from_csv(csv_file, cfg, optimizer_order, steps=None, out_dir=None):
    """
    Read the results CSV and save validation perplexity/loss plots.
    """
    if out_dir is None:
        out_dir = os.path.dirname(csv_file) or "."
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_file):
        print(f"[Warning] CSV not found, skip plotting: {csv_file}")
        return

    mpl.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 16,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
    })

    display_name = {
        "AdamW": "AdamW",
        "Adam": "Adam",
        "AdamW_lr": "AdamW with lr schedule",
        "Adam_lr": "Adam with lr schedule",
        "AdamSHANG": "Adam-SHANG",
        "AdamSHANGs": "Adam-SHANG-s",
        "AdamSHANG(beta=0,gamma=1)": r"Adam-SHANG ($\beta=0,\gamma=1$)",
        "AdamSHANGs(beta=0,gamma=1)": r"Adam-SHANG-s ($\beta=0,\gamma=1$)",
    }

    color_map = {
        "AdamW": "blue",
        "Adam": "green",
        "AdamW_lr": "deepskyblue",
        "Adam_lr": "magenta",
        "AdamSHANGs": "orange",
        "AdamSHANG": "red",
        "AdamSHANG(beta=0,gamma=1)": "brown",
        "AdamSHANGs(beta=0,gamma=1)": "gray",
    }

    raw = defaultdict(lambda: defaultdict(list))

    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalize_name(row["optimizer"])
            seed = row["seed"]
            raw[name][seed].append({
                "step": int(row["step"]),
                "val_loss": float(row["val_loss"]),
                "val_ppl": float(row["val_ppl"]),
            })

    # Deduplicate each optimizer/seed trajectory by step.
    for name in raw:
        for seed in raw[name]:
            rows = sorted(raw[name][seed], key=lambda r: r["step"])
            dedup = {}
            for r in rows:
                dedup[r["step"]] = r
            raw[name][seed] = {k: dedup[k] for k in sorted(dedup)}

    data = {}
    for name in optimizer_order:
        name = normalize_name(name)
        if name not in raw:
            print(f"[Warning] No rows found for {name}; skip in plot.")
            continue

        seeds = list(raw[name].keys())
        if not seeds:
            continue

        common_steps = set(raw[name][seeds[0]].keys())
        for seed in seeds[1:]:
            common_steps &= set(raw[name][seed].keys())
        common_steps = sorted(common_steps)

        if steps is not None:
            common_steps = [s for s in common_steps if s <= steps]
        if not common_steps:
            print(f"[Warning] No common evaluation steps for {name}; skip in plot.")
            continue

        ppl_mat = np.array([
            [raw[name][seed][s]["val_ppl"] for s in common_steps]
            for seed in seeds
        ])
        vl_mat = np.array([
            [raw[name][seed][s]["val_loss"] for s in common_steps]
            for seed in seeds
        ])

        data[name] = {
            "step": np.array(common_steps),
            "mean_val_ppl": ppl_mat.mean(axis=0),
            "std_val_ppl": ppl_mat.std(axis=0),
            "mean_val_loss": vl_mat.mean(axis=0),
            "std_val_loss": vl_mat.std(axis=0),
        }

    if not data:
        print("[Warning] No valid data found for plotting.")
        return

    step_suffix = f"_first{steps}steps" if steps else ""
    model_tag = f"{cfg['n_layers']}L_d{cfg['d_model']}"
    prefix = f"text8_Transformer_{model_tag}_batch{cfg['batch_size']}"

    def savefig(fname):
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {path}")

    # Validation perplexity.
    plt.figure(figsize=(7, 5))
    for name in optimizer_order:
        name = normalize_name(name)
        if name not in data:
            continue
        d = data[name]
        x = d["step"]
        mean = d["mean_val_ppl"]
        std = d["std_val_ppl"]
        color = color_map.get(name, "black")
        label = display_name.get(name, name)
        plt.plot(x, mean, label=label, color=color, linewidth=2.5)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        print(f"{name:>32}  final ppl = {mean[-1]:.2f} +/- {std[-1]:.2f}")

    plt.xlabel("Training steps")
    plt.ylabel("Validation Perplexity $\\downarrow$")
    plt.title(f"text8 Layers = {cfg['n_layers']} (batch={cfg['batch_size']}) Validation Perplexity")
    plt.legend(loc="upper right")
    plt.tight_layout()
    savefig(prefix + f"_Val_Perplexity{step_suffix}.png")

    # Validation loss on a log scale.
    plt.figure(figsize=(7, 5))
    for name in optimizer_order:
        name = normalize_name(name)
        if name not in data:
            continue
        d = data[name]
        x = d["step"]
        mean = np.clip(d["mean_val_loss"], 1e-8, None)
        std = d["std_val_loss"]
        color = color_map.get(name, "black")
        label = display_name.get(name, name)
        plt.semilogy(x, mean, label=label, color=color, linewidth=2.5)
        plt.fill_between(
            x,
            np.clip(mean - std, 1e-8, None),
            mean + std,
            alpha=0.2,
            color=color,
        )

    ax = plt.gca()
    yticks = [1.0, 1.2, 1.4, 1.6, 1.8]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.ticklabel_format(style="plain", axis="y")

    plt.xlabel("Training steps")
    plt.ylabel("Loss (log scale)")
    plt.title(f"text8 Layers = {cfg['n_layers']} (batch={cfg['batch_size']}) Validation Loss")
    plt.legend(loc="upper right")
    plt.tight_layout()
    savefig(prefix + f"_Val_Loss{step_suffix}.png")

# Main experiment
OPTIMIZERS = [
    "AdamSHANG",
    "AdamSHANGs",
    "Adam",
    "AdamW",
    "Adam_lr",
    "AdamW_lr",
]
ABLATIONS = ["AdamSHANG(beta=0,gamma=1)", "AdamSHANGs(beta=0,gamma=1)"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=CFG["total_steps"])
    parser.add_argument("--batch-size", type=int, default=CFG["batch_size"])
    parser.add_argument("--seeds", type=int, nargs="+", default=CFG["seeds"])
    parser.add_argument("--data-path", default=CFG["data_path"])
    parser.add_argument("--download", action="store_true", help="Download text8 if data-path does not exist.")
    parser.add_argument("--output-dir", default=CFG["output_dir"])
    parser.add_argument("--eval-every", type=int, default=CFG["eval_every"])
    parser.add_argument("--eval-batches", type=int, default=CFG["eval_batches"])
    parser.add_argument("--log-every", type=int, default=CFG["log_every"])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--optimizers", choices=OPTIMIZERS, nargs="+", default=None)
    parser.add_argument("--ablation", action="store_true", help="Also run the two beta=0,gamma=1 variants.")
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.eval_every, args.eval_batches, args.log_every) <= 0:
        parser.error("Training and evaluation counts must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be distinct nonnegative integers.")
    return args


def main(argv=None):
    global DEVICE
    args = parse_args(argv)
    DEVICE = torch.device(args.device) if args.device else get_device()
    cfg = dict(CFG, total_steps=args.steps, batch_size=args.batch_size, seeds=args.seeds,
               data_path=args.data_path, download=args.download, output_dir=args.output_dir,
               eval_every=args.eval_every, eval_batches=args.eval_batches, log_every=args.log_every)
    optimizers = list(args.optimizers or OPTIMIZERS)
    if args.ablation:
        optimizers.extend(ABLATIONS)
    print(f"Device: {DEVICE}")
    print(f"\n{'★'*60}")
    print(f"  text8 full  |  optimizers={optimizers}")
    print(f"  scheduled baselines={sorted(SCHEDULED_OPTIMIZERS.intersection(optimizers))} | warmup={cfg['warmup_steps']}")
    print(f"{'★'*60}")

    train_loader, val_loader, vocab_size = build_data(cfg)
    print(f"\n  batch={cfg['batch_size']}  seq_len={cfg['seq_len']}  "
          f"steps={cfg['total_steps']}  layers={cfg['n_layers']}")

    out_dir = os.path.join(cfg["output_dir"], "text8_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    os.makedirs(out_dir, exist_ok=False)
    csv_path = os.path.join(out_dir, f"Layer{cfg['n_layers']}_Batch{cfg['batch_size']}_fulltext8_results.csv")
    with open(os.path.join(out_dir, "config.json"), "w") as handle:
        json.dump({"config": cfg, "optimizers": optimizers, "device": str(DEVICE),
                   "torch_version": torch.__version__}, handle, indent=2)
    seeds = cfg["seeds"]
    seed_results = {name: [] for name in optimizers}

    write_header = not os.path.exists(csv_path)
    csv_file   = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["optimizer", "seed", "step",
                             "train_loss", "val_loss", "val_ppl"])
        csv_file.flush()

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'★'*60}")
        print(f"  Seed {seed_idx+1}/{len(seeds)}  (seed={seed})")
        print(f"{'★'*60}")

        for name in optimizers:
            hist = run(name, train_loader, val_loader, vocab_size, cfg, seed=seed)
            seed_results[name].append(hist)

            if hist["val_loss"]:
                print(
                    f"  Final {name} | seed={seed} | "
                    f"train_loss={hist['train_loss'][-1]:.4f} | "
                    f"val_loss={hist['val_loss'][-1]:.4f} | "
                    f"val_ppl={hist['val_ppl'][-1]:.2f}"
                )

            for s, tl, vl, vp in zip(hist["step"], hist["train_loss"],
                                      hist["val_loss"],  hist["val_ppl"]):
                csv_writer.writerow([name, seed, s, tl, vl, vp])
            csv_file.flush()
            print(f"  ✓ {name} seed={seed} → {csv_path}")

    csv_file.close()

    # Print summary statistics across seeds.
    import numpy as np
    all_results = {}
    for name in optimizers:
        runs = seed_results[name]
        if len(runs) == 0 or len(runs[0]["step"]) == 0:
            print(f"[Warning] No validation records for {name}; skip summary.")
            continue
        steps = runs[0]["step"]
        all_results[name] = {"step": steps}
        for key in ["train_loss", "val_loss", "val_ppl"]:
            matrix = np.array([r[key] for r in runs])
            all_results[name][f"mean_{key}"] = matrix.mean(axis=0)
            all_results[name][f"std_{key}"]  = matrix.std(axis=0)

    print(f"\n{'═'*70}")
    print(f"  Final results @ step {cfg['total_steps']}  "
          f"({len(seeds)} seeds, full text8)")
    print(f"{'═'*70}")
    print(f"  {'Optimizer':<22} {'lr':>8} "
          f"{'Val Loss (mean±std)':>22} {'PPL (mean±std)':>20}")
    print(f"  {'─'*68}")
    for name in all_results:
        r   = all_results[name]
        vl  = r["mean_val_loss"][-1];  vl_s = r["std_val_loss"][-1]
        vp  = r["mean_val_ppl"][-1];   vp_s = r["std_val_ppl"][-1]
        print(f"  {name:<22} {cfg['lr'][name]:>8.2e} "
              f"{vl:>10.4f} ± {vl_s:<8.4f} "
              f"{vp:>8.2f} ± {vp_s:<6.2f}")

    print(f"\n  All results saved -> {csv_path}")

    # Generate validation loss and perplexity figures from the saved CSV.
    plot_results_from_csv(csv_path, cfg, optimizers)


if __name__ == "__main__":
    main()
