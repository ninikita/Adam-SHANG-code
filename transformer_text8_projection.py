"""
Text8 character-level language modeling: auxiliary-variable projection ablation.

This standalone experiment compares Adam-SHANG and Adam-SHANG-s with their
projected counterparts. Only the auxiliary variable y is projected, immediately
after its update, coordinatewise onto [-R/2, R/2]. The model parameters x are
not projected. R=10 reproduces clipping to [-5, 5]. The optimizer module keeps
the original preconditioner/update ordering for each method.

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
    - batch sizes = 32 and 128
    - sequence length = 256
    - total training steps = 30000
    - gradient clipping threshold = 1.0
    - validation is performed every 2000 steps
    - each validation uses 100 batches
    - all four methods use their own stepsizes, with no external scheduler

Reproducibility:
    - The random seeds are specified in CFG["seeds"].
    - Each projected/control pair shares all hyperparameters and resets the
      same random seed before model initialization and shuffled data iteration.
    - The model, dataset split, training/evaluation order, and hyperparameters
      match transformer_text8.py; the original script is not imported.
    - Each invocation writes a new timestamped output directory. Every CSV
      trajectory and per-seed summary identifies batch size, seed, projection
      flag, R, coordinate bound R/2, and all optimizer parameters.
    - A JSON configuration and a mean/std summary across seeds are also saved.
    - Optimizer errors stop the experiment; failed updates are not silently
      counted as completed training steps.
    - The script reports mean and standard deviation across seeds at the end.
    - The exact optimizer hyperparameters are specified in CFG.

Compute:
    - The script supports CUDA, Apple MPS, and CPU.
    - Run with --help for data, output, seed, and training options.
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
from AdamSHANG import AdamSHANG, AdamSHANGs, AdamSHANGProjected, AdamSHANGsProjected

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

    batch_sizes   = [32, 128],
    total_steps   = 30000,
    log_every     = 500,
    eval_every    = 2000,
    eval_batches  = 100,
    grad_clip     = 1.0,

    lr = {
        "AdamSHANG":          0.5,
        "AdamSHANGs":         0.5,
    },

    # Explicitly include the defaults used by transformer_text8.py. Each pair
    # uses one shared parameter dictionary; only the projected class receives R.
    adamshang  = dict(beta=0.05, gamma=0.001, weight_decay=1e-2,
                     init_P=1, eps=1e-6, alpha_clip=0.99),
    adamshangs = dict(beta=0.05, gamma=0.01, weight_decay=1e-2,
                     init_P=1, eps=1e-10, alpha_clip=0.99),

    seeds      = [23,24,25],
    projection_R = 10.0,
    output_dir = "results/text8_projection",
    data_path  = "data/text8.txt",
    download   = False,
)

OPTIMIZERS = ["AdamSHANG", "AdamSHANGProjected",
              "AdamSHANGs", "AdamSHANGsProjected"]

# Name -> (optimizer class, unprojected family, configuration key, projection).
OPTIMIZER_SPECS = {
    "AdamSHANG": (AdamSHANG, "AdamSHANG", "adamshang", False),
    "AdamSHANGProjected": (AdamSHANGProjected, "AdamSHANG", "adamshang", True),
    "AdamSHANGs": (AdamSHANGs, "AdamSHANGs", "adamshangs", False),
    "AdamSHANGsProjected": (AdamSHANGsProjected, "AdamSHANGs", "adamshangs", True),
}

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

_TOKEN_CACHE = None
_TOKEN_CACHE_PATH = None

def build_data(cfg):
    global _TOKEN_CACHE, _TOKEN_CACHE_PATH
    data_path = cfg.get("data_path", "data/text8.txt")
    absolute_path = os.path.abspath(data_path)
    if _TOKEN_CACHE is None or _TOKEN_CACHE_PATH != absolute_path:
        if not os.path.isfile(data_path):
            if not cfg.get("download", False):
                raise FileNotFoundError(f"{data_path} not found. Supply --data-path or allow --download.")
            download_text8(data_path)
        print("Reading full text8 (~100M chars)...")
        with open(data_path, encoding="utf-8") as f:
            text = f.read()
        print(f"  Total chars: {len(text):,}")

        vocab    = sorted(set(text))
        char2idx = {c: i for i, c in enumerate(vocab)}
        vocab_size = len(vocab)
        print(f"  Vocab size: {vocab_size}")

        encoded = torch.tensor([char2idx[c] for c in text], dtype=torch.long)
        split = int(len(encoded) * 0.9)
        _TOKEN_CACHE = (encoded[:split], encoded[split:], vocab_size)
        _TOKEN_CACHE_PATH = absolute_path

    # Cache tokens, not loaders: both batch sizes must use their own loaders.
    train_tok, val_tok, vocab_size = _TOKEN_CACHE
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
    return train_dl, val_dl, vocab_size

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
    if name not in OPTIMIZER_SPECS:
        raise ValueError(f"Unknown optimizer: {name}")
    cls, family, key, projected = OPTIMIZER_SPECS[name]
    kwargs = dict(cfg[key])
    if projected:
        kwargs["R"] = cfg["projection_R"]
    return cls(params, lr=cfg["lr"][family], **kwargs)


def optimizer_metadata(name, cfg, seed):
    _, family, key, projected = OPTIMIZER_SPECS[name]
    params = dict(lr=cfg["lr"][family], **cfg[key])
    return dict(optimizer=name, base_optimizer=family,
                batch_size=cfg["batch_size"], seed=seed,
                projected=projected, projection_R=cfg["projection_R"],
                projection_bound=cfg["projection_R"] / 2.0,
                **params,
                init_alpha_prev=params["lr"] * math.sqrt(params["init_P"]))


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
    metadata = optimizer_metadata(opt_name, cfg, seed)
    print(f"\n{'─'*60}")
    print(f"  {opt_name}   lr={metadata['lr']}  seed={seed}  "
          f"batch={cfg['batch_size']}  projected={metadata['projected']}  "
          f"R={cfg['projection_R']}")
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
        except RuntimeError as e:
            raise RuntimeError(
                f"{opt_name}, batch={cfg['batch_size']}, seed={seed}, "
                f"step={step}: optimizer update failed; experiment stopped."
            ) from e
        history["step_times"].append(time.perf_counter() - t_step)

        if step % cfg["log_every"] == 0:
            elapsed    = time.time() - t0
            current_lr = opt.param_groups[0]["lr"]
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
    return name.strip()


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
        "AdamSHANG": "Adam-SHANG (unprojected)",
        "AdamSHANGs": "Adam-SHANG-s (unprojected)",
        "AdamSHANGProjected": "Adam-SHANG (projected)",
        "AdamSHANGsProjected": "Adam-SHANG-s (projected)",
    }

    color_map = {
        "AdamSHANGs": "orange",
        "AdamSHANG": "red",
        "AdamSHANGProjected": "blue",
        "AdamSHANGsProjected": "green",
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
    prefix = (f"text8_Transformer_{model_tag}_batch{cfg['batch_size']}"
              f"_projection_R{cfg['projection_R']:g}")

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
    plt.title(f"text8, batch={cfg['batch_size']}, R={cfg['projection_R']:g}: Validation Perplexity")
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
    plt.title(f"text8, batch={cfg['batch_size']}, R={cfg['projection_R']:g}: Validation Loss")
    plt.legend(loc="upper right")
    plt.tight_layout()
    savefig(prefix + f"_Val_Loss{step_suffix}.png")

# Main experiment
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=CFG["total_steps"])
    parser.add_argument("--batch-sizes", "--batch-size", type=int, nargs="+", default=CFG["batch_sizes"])
    parser.add_argument("--seeds", type=int, nargs="+", default=CFG["seeds"])
    parser.add_argument("--data-path", default=CFG["data_path"])
    parser.add_argument("--download", action="store_true", help="Download text8 if data-path does not exist.")
    parser.add_argument("--output-dir", default=CFG["output_dir"])
    parser.add_argument("--eval-every", type=int, default=CFG["eval_every"])
    parser.add_argument("--eval-batches", type=int, default=CFG["eval_batches"])
    parser.add_argument("--log-every", type=int, default=CFG["log_every"])
    parser.add_argument("--projection-r", type=float, default=CFG["projection_R"])
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--optimizers", choices=OPTIMIZERS, nargs="+", default=None)
    args = parser.parse_args(argv)
    if min(args.steps, args.eval_every, args.eval_batches, args.log_every, *args.batch_sizes) <= 0:
        parser.error("Training and evaluation counts must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be distinct nonnegative integers.")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("batch-sizes must be distinct.")
    if not math.isfinite(args.projection_r) or args.projection_r <= 0:
        parser.error("projection-r must be finite and positive.")
    return args


def main(argv=None):
    global DEVICE
    args = parse_args(argv)
    DEVICE = torch.device(args.device) if args.device else get_device()
    config = dict(CFG, total_steps=args.steps, batch_sizes=args.batch_sizes, seeds=args.seeds,
                  data_path=args.data_path, download=args.download, output_dir=args.output_dir,
                  eval_every=args.eval_every, eval_batches=args.eval_batches, log_every=args.log_every,
                  projection_R=args.projection_r)
    optimizers = list(args.optimizers or OPTIMIZERS)
    if not math.isfinite(config["projection_R"]) or config["projection_R"] <= 0:
        raise ValueError("projection_R must be a finite positive number.")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be nonempty and contain distinct values.")
    if (not config["batch_sizes"] or any(b <= 0 for b in config["batch_sizes"])
            or len(set(config["batch_sizes"])) != len(config["batch_sizes"])):
        raise ValueError("batch_sizes must contain distinct positive values.")
    if any(config[key] <= 0 for key in ("total_steps", "log_every", "eval_every", "eval_batches")):
        raise ValueError("Training and evaluation step counts must be positive.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = os.path.join(config["output_dir"], f"text8_projection_{run_id}")
    os.makedirs(out_dir, exist_ok=False)
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "config": config,
            "optimizers": optimizers,
            "device": str(DEVICE),
            "torch_version": torch.__version__,
            "projection_applied_to": "y only, immediately after the y update",
            "projection_box": "[-projection_R/2, projection_R/2] coordinatewise",
            "control_projection": "disabled; R and bound record the paired experiment setting",
            "train_loss_timing": "before optimizer step (same as transformer_text8.py)",
            "validation_timing": "after optimizer step (same as transformer_text8.py)",
            "external_scheduler": None,
            "optimizer_error_policy": "fail fast",
            "std_ddof": 0,
        }, f, indent=2)

    print(f"\n{'★'*60}")
    print(f"  text8 projection comparison  |  optimizers={optimizers}")
    print(f"  batches={config['batch_sizes']}  seeds={config['seeds']}  R={config['projection_R']}")
    print("  No external scheduler; only y is projected, never x.")
    print(f"  Results -> {out_dir}")
    print(f"{'★'*60}")

    seeds = config["seeds"]
    for batch_size in config["batch_sizes"]:
        cfg = dict(config, batch_size=batch_size)
        train_loader, val_loader, vocab_size = build_data(cfg)
        print(f"\n  batch={batch_size}  seq_len={cfg['seq_len']}  "
              f"steps={cfg['total_steps']}  layers={cfg['n_layers']}")

        prefix = (f"Layer{cfg['n_layers']}_Batch{batch_size}_fulltext8"
                  f"_paired_projection_R{cfg['projection_R']:g}")
        csv_path = os.path.join(out_dir, prefix + "_trajectories.csv")
        per_seed_path = os.path.join(out_dir, prefix + "_per_seed.csv")
        summary_path = os.path.join(out_dir, prefix + "_summary.csv")
        metadata_fields = list(optimizer_metadata(optimizers[0], cfg, seeds[0]))
        result_fields = metadata_fields + ["step", "train_loss", "val_loss", "val_ppl"]
        seed_results = {name: [] for name in optimizers}

        with open(csv_path, "w", newline="") as csv_file, \
                open(per_seed_path, "w", newline="") as per_seed_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=result_fields)
            final_writer = csv.DictWriter(per_seed_file, fieldnames=result_fields)
            csv_writer.writeheader()
            final_writer.writeheader()
            csv_file.flush()
            per_seed_file.flush()

            for seed_idx, seed in enumerate(seeds):
                print(f"\n{'★'*60}")
                print(f"  batch={batch_size}  Seed {seed_idx+1}/{len(seeds)} (seed={seed})")
                print(f"{'★'*60}")

                for name in optimizers:
                    hist = run(name, train_loader, val_loader, vocab_size, cfg, seed=seed)
                    if not hist["step"] or hist["step"][-1] != cfg["total_steps"]:
                        raise RuntimeError(f"Incomplete training history: {name}, batch={batch_size}, seed={seed}")
                    seed_results[name].append(hist)
                    metadata = optimizer_metadata(name, cfg, seed)

                    for s, tl, vl, vp in zip(hist["step"], hist["train_loss"],
                                              hist["val_loss"], hist["val_ppl"]):
                        row = dict(metadata, step=s, train_loss=tl, val_loss=vl, val_ppl=vp)
                        csv_writer.writerow(row)
                    final_writer.writerow(row)
                    csv_file.flush()
                    per_seed_file.flush()
                    print(f"  Final {name} | batch={batch_size} | seed={seed} | "
                          f"val_loss={hist['val_loss'][-1]:.4f} | val_ppl={hist['val_ppl'][-1]:.2f}")

        # Final-step mean/std, with exact seed membership and optimizer metadata.
        summary_rows = []
        print(f"\n{'═'*70}")
        print(f"  Final results @ step {cfg['total_steps']}  batch={batch_size}  "
              f"R={cfg['projection_R']:g}  ({len(seeds)} seeds)")
        print(f"{'═'*70}")
        for name in optimizers:
            runs = seed_results[name]
            row = optimizer_metadata(name, cfg, seed=None)
            del row["seed"]
            row.update(seeds=json.dumps(seeds), n_seeds=len(runs), step=cfg["total_steps"])
            for key in ["train_loss", "val_loss", "val_ppl"]:
                values = np.array([hist[key][-1] for hist in runs])
                row[f"mean_{key}"] = values.mean()
                row[f"std_{key}"] = values.std()
            summary_rows.append(row)
            print(f"  {name:<24} n={row['n_seeds']}  "
                  f"val_loss={row['mean_val_loss']:.4f} ± {row['std_val_loss']:.4f}  "
                  f"ppl={row['mean_val_ppl']:.2f} ± {row['std_val_ppl']:.2f}")

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\n  Trajectories -> {csv_path}")
        print(f"  Per-seed results -> {per_seed_path}")
        print(f"  Across-seed summary -> {summary_path}")
        plot_results_from_csv(csv_path, cfg, optimizers, out_dir=out_dir)


if __name__ == "__main__":
    main()
