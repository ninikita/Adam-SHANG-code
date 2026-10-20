"""
Fixed-learning-rate (beta, gamma) grid search on CIFAR-100/ResNet-50.

Defaults preserve the existing 28-configuration training-loss sweep. This is
not an automatic staged search, and it does not search Adam/AdamW settings.
CIFAR-10/ResNet-34 and the lagged method can be selected explicitly.
Dataset downloads require --download and are disabled by default.
Importing the module does not download data, train models, or write files.
"""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset

from AdamSHANG import AdamSHANG, AdamSHANGs


GAMMAS = [5e-3, 1e-3, 5e-2, 1e-2, 5e-1, 1e-1, 1]
BETAS = [5e-2, 1e-2, 5e-1, 1e-1]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loader(args):
    # One subset is sampled after setting the global NumPy seed, as before.
    # All grid entries use that subset; no test-set selection is performed.
    if args.dataset == "CIFAR100":
        dataset_cls = datasets.CIFAR100
        mean, std = (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)
    else:
        dataset_cls = datasets.CIFAR10
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = dataset_cls(args.data_dir, train=True, download=args.download,
                          transform=transform)
    if args.subset_size:
        indices = np.random.permutation(len(dataset))[:args.subset_size]
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                      num_workers=0, pin_memory=False)


def build_model(dataset, device):
    # As in the original sweep, each configuration gets a fresh model without
    # resetting the global seed between configurations.
    if dataset == "CIFAR100":
        model = models.resnet50(weights=None)
        classes = 100
    else:
        model = models.resnet34(weights=None)
        classes = 10
    model.fc = nn.Linear(model.fc.in_features, classes, bias=True)
    return model.to(device)


def train_one_config(beta, gamma, train_loader, device, args):
    model = build_model(args.dataset, device)
    optimizer_cls = AdamSHANGs if args.optimizer == "AdamSHANGs" else AdamSHANG
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, beta=beta,
                              gamma=gamma, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    epoch_losses = []

    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0.0
        print(f"lr={args.lr:g}, beta={beta}, gamma={gamma}, "
              f"epoch={epoch + 1}/{args.epochs}")
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        # Preserve the original unweighted mean over minibatch losses.
        epoch_losses.append(avg_loss / len(train_loader))

    del model, optimizer
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return epoch_losses


def plot_losses(losses, args):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_gamma = {
        1e-1: "tab:red", 5e-1: "tab:blue", 1e-2: "tab:green",
        5e-2: "tab:pink", 5: "tab:cyan", 1: "tab:olive",
        1e-3: "tab:purple", 2: "tab:brown", 5e-3: "tab:gray",
    }
    style_beta = {
        0.1: "solid", 0.5: "dashed", 1e-2: "dotted", 5e-2: "dashdot",
        1e-3: (0, (5, 1)), 5e-3: (0, (3, 1, 1, 1)),
    }
    plot_lines = {}
    plt.figure(figsize=(10, 6))
    for beta in BETAS:
        for gamma in GAMMAS:
            if (beta, gamma) not in losses:
                continue
            line, = plt.semilogy(
                range(1, args.epochs + 1), losses[(beta, gamma)],
                color=color_gamma.get(gamma, "black"),
                linestyle=style_beta.get(beta, "solid"), linewidth=2,
            )
            plot_lines[(beta, gamma)] = line
    beta_lines = [(beta, next(line for (b, _), line in plot_lines.items()
                             if b == beta))
                  for beta in BETAS if any(b == beta for b, _ in plot_lines)]
    gamma_lines = [(gamma, next(line for (_, g), line in plot_lines.items()
                               if g == gamma))
                   for gamma in GAMMAS if any(g == gamma for _, g in plot_lines)]
    legend1 = plt.legend([line for _, line in beta_lines],
                         [f"beta={beta}" for beta, _ in beta_lines],
                         loc="upper right", title="Beta (Linestyle)")
    plt.gca().add_artist(legend1)
    plt.legend([line for _, line in gamma_lines],
               [f"gamma={gamma}" for gamma, _ in gamma_lines],
               loc="lower left", title="Gamma (Color)")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"Grid Search over (lr={args.lr:g}, beta, gamma) "
              f"for {args.optimizer} on {args.dataset}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    lr_tag = f"{args.lr:g}".replace(".", "p")
    plot_path = Path(args.output_dir) / (
        f"{args.optimizer}_lr{lr_tag}_beta_gamma_{args.dataset}_grid_search_plot.pdf"
    )
    plt.savefig(plot_path)
    if args.show:
        plt.show()
    plt.close()
    print(f"Saved plot to {plot_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fixed-lr CIFAR (beta, gamma) training-loss sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["CIFAR100", "CIFAR10"], default="CIFAR100")
    parser.add_argument("--optimizer", choices=["AdamSHANGs", "AdamSHANG"],
                        default="AdamSHANGs")
    parser.add_argument("--data-dir", default="data/cifar")
    parser.add_argument("--download", action="store_true",
                        help="Allow torchvision to download the selected CIFAR dataset.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-size", type=int, default=20000,
                        help="Training images used; 0 uses the full training set.")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Fixed learning rate, not a second-stage search.")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"],
                        default="auto")
    parser.add_argument("--max-configs", type=int, default=None,
                        help="Use only the first N grid entries; not the complete search.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the grid and budget without downloading or training.")
    parser.add_argument("--show", action="store_true", help="Show the saved plot interactively.")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Epochs, batch size, and lr must be positive and finite.")
    if args.subset_size < 0 or args.seed < 0:
        parser.error("Subset size and seed must be nonnegative.")
    if args.max_configs is not None and args.max_configs < 1:
        parser.error("--max-configs must be positive.")
    return args


def main(argv=None):
    args = parse_args(argv)
    combinations = [(beta, gamma) for beta in BETAS for gamma in GAMMAS]
    if args.max_configs is not None:
        combinations = combinations[:args.max_configs]
    print(f"{args.dataset}, {args.optimizer}, lr={args.lr:g}: "
          f"{len(combinations)}/{len(BETAS) * len(GAMMAS)} pairs "
          f"x {args.epochs} epochs, batch={args.batch_size}.")
    print(f"beta={BETAS}; gamma={GAMMAS}")
    if args.max_configs:
        print("Limited-grid run: this does not reproduce the full hyperparameter search.")
    if args.dry_run:
        return

    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Training on device: {device}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    train_loader = build_loader(args)
    losses = {}
    for beta, gamma in combinations:
        losses[(beta, gamma)] = train_one_config(beta, gamma, train_loader, device, args)
        print(f"  beta={beta}, gamma={gamma} -> final_loss={losses[(beta, gamma)][-1]:.4f}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(args.output_dir) / f"{args.optimizer}_losses_from_grid_search.pkl"
    with save_path.open("wb") as handle:
        pickle.dump(losses, handle)
    print(f"Saved losses to {save_path}")
    plot_losses(losses, args)


if __name__ == "__main__":
    main()
