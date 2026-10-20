"""
CIFAR-100 image classification experiment with ResNet-50.

This script compares Adam-SHANG and Adam-SHANG-s with Adam, AdamW,
and SHANG++ on CIFAR-100 using a ResNet-50 model initialized from scratch.

Dataset:
    - CIFAR-100 from torchvision.datasets.CIFAR100.
    - Training split: the official CIFAR-100 training set.
    - Test split: the official CIFAR-100 test set.
    - Standard CIFAR augmentation is used for training: random crop with
      padding 4 and random horizontal flip.
    - Test images use only normalization.

Model:
    - torchvision.models.resnet50(weights=None)
    - The final fully connected layer is replaced by a 100-class classifier.

Training protocol:
    - batch size = 128/32
    - number of epochs = 50
    - test evaluation every 5 epochs
    - 3 independent seeds: 23, 24, 25
    - no external learning-rate schedule is used in this script

Reproducibility:
    - The random seeds are set before each run.
    - Each optimizer receives the same model initialization and DataLoader
      shuffle order for a given seed.

Compute:
    - The script supports CUDA, Apple MPS, and CPU.
    - Run with --help for data, output, seed, and training options.
    - Dataset downloads require --download. Add --ablation for beta=0,gamma=1.
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, models, transforms

from AdamSHANG import AdamSHANG, AdamSHANGs
from SHANG_PlusPlus import SHANG_PlusPlus
from util import ProgressBar


class Trainer:
    def __init__(self, model, opt_name: str, data_dir="data/cifar",
                 download=False, device=None):
        self.data_dir = data_dir
        self.download = download
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.net = model.to(self.device)
        self.optimizer = None
        self.scheduler = None

        self.train_accuracies = []
        self.train_losses = []
        self.test_accuracies = []
        self.test_losses = []
        self.start_epoch = 0

        exec('self.optimizer =' + opt_name)

    def train(
        self,
        save_dir,
        num_epochs=30,
        batch_size=50,
        seed=0,
        schedule_lr_epochs=0,
        lr_factor1=1.0,
        lr_factor2=1.0,
        test_every=5,
        verbose=False,
        manual_seed=False,
        num_workers=0,
    ):
        print(f"Training on device: {self.device}")
        if manual_seed:
            torch.manual_seed(0)

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)
            ),
        ])

        train_dataset = datasets.CIFAR100(
            self.data_dir,
            train=True,
            download=self.download,
            transform=train_transform
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
            generator=generator,
        )

        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        progress_bar = ProgressBar()

        if self.start_epoch == 0:
            test_loss, test_accuracy = self.test(batch_size=batch_size, num_workers=num_workers)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)

        for epoch in range(self.start_epoch + 1, num_epochs + 1):
            print(f'Epoch {epoch}/{num_epochs}')
            self.net.train()

            if schedule_lr_epochs and epoch % schedule_lr_epochs == 0:
                for g in self.optimizer.param_groups:
                    if 'lr' in g:
                        g['lr'] *= lr_factor1
                    if 'time_scale' in g:
                        g['time_scale'] *= lr_factor2

            for i, data in enumerate(data_loader, 1):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                _, predicted = torch.max(outputs.data, dim=1)
                batch_total = labels.size(0)
                batch_correct = (predicted == labels).sum().item()

                self.train_accuracies.append(batch_correct / batch_total)
                self.train_losses.append(loss.item())

                if verbose:
                    info_str = f'Last batch accuracy: {batch_correct / batch_total:.4f}'
                    progress_bar.update(
                        max_value=len(data_loader),
                        current_value=i,
                        info=info_str
                    )

            if verbose:
                progress_bar.new_line()

            if epoch % test_every == 0 or epoch == num_epochs:
                test_loss, test_accuracy = self.test(batch_size=250, num_workers=num_workers)
                self.test_losses.append(test_loss)
                self.test_accuracies.append(test_accuracy)
                if verbose:
                    print(f'Test accuracy: {test_accuracy:.4f}')

            if epoch % 10 == 0 or epoch == num_epochs:
                self.save_parameters(epoch, directory=save_dir)

    def test(self, batch_size=250, num_workers=0):
        self.net.eval()

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)
            ),
        ])

        test_dataset = datasets.CIFAR100(
            self.data_dir,
            train=False,
            download=self.download,
            transform=test_transform
        )

        data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        correct = 0
        total = 0
        loss_sum = 0.0

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss_sum += loss.item()

        self.net.train()
        return (loss_sum / (i + 1), correct / total)

    def save_parameters(self, epoch, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            'train_accuracies': self.train_accuracies,
            'train_losses': self.train_losses,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses
        }, os.path.join(directory, f'checkpoint_{epoch}.pth'))

    def load_parameters(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_state = checkpoint.get('scheduler_state_dict')
        if self.scheduler is not None and scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)
        self.train_accuracies = checkpoint['train_accuracies']
        self.train_losses = checkpoint['train_losses']
        self.test_accuracies = checkpoint['test_accuracies']
        self.test_losses = checkpoint['test_losses']
        self.start_epoch = checkpoint['epoch']


# =========================================================
# Main experiment
# =========================================================

OPTIMIZERS = {
    'Adam-SHANG': 'AdamSHANG(self.net.parameters(), lr=0.1, beta=0.1, gamma=0.005, weight_decay=1e-2)',
    'Adam-SHANG-s': 'AdamSHANGs(self.net.parameters(), lr=0.1, beta=0.05, gamma=0.001, weight_decay=1e-2)',
    'Adam': 'torch.optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)',
    'AdamW': 'torch.optim.AdamW(self.net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)',
    'SHANG++': 'SHANG_PlusPlus(self.net.parameters(), lr=0.5, time_scale=10, rho=1.5, weight_decay=1e-5)',
}
ABLATIONS = {
    'Adam-SHANG(beta=0,gamma=1)': 'AdamSHANG(self.net.parameters(), lr=0.1, beta=0, gamma=1, weight_decay=1e-2)',
    'Adam-SHANG-s(beta=0,gamma=1)': 'AdamSHANGs(self.net.parameters(), lr=0.1, beta=0, gamma=1, weight_decay=1e-2)',
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[23, 24, 25])
    parser.add_argument("--data-dir", default="data/cifar")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-100.")
    parser.add_argument("--output-dir", default="results/cifar100")
    parser.add_argument("--test-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--optimizers", nargs="+", choices=list(OPTIMIZERS), default=None)
    parser.add_argument("--ablation", action="store_true", help="Also run the two beta=0,gamma=1 variants.")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.test_every) <= 0 or args.num_workers < 0:
        parser.error("epochs, batch-size and test-every must be positive; num-workers must be nonnegative.")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("seeds must be distinct integers in [0, 2**32).")
    return args


def main(argv=None):
    args = parse_args(argv)
    opt_names = {name: OPTIMIZERS[name] for name in (args.optimizers or OPTIMIZERS)}
    if args.ablation:
        opt_names.update(ABLATIONS)
    for seed in args.seeds:
        print(f"\nSeed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        for key, opt_name in opt_names.items():
            print(f"Training optimizer: {key}")
            torch.manual_seed(seed)
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 100, bias=True)
            net = Trainer(model=model, opt_name=opt_name, data_dir=args.data_dir,
                          download=args.download, device=args.device)
            net.train(
                save_dir=os.path.join(args.output_dir, f'CIFAR100-ResNet50_batch{args.batch_size}_{key}', f'seed{seed}'),
                batch_size=args.batch_size, num_epochs=args.epochs, seed=seed,
                schedule_lr_epochs=0, lr_factor1=1.0, lr_factor2=1.0,
                test_every=args.test_every, manual_seed=False, verbose=False,
                num_workers=args.num_workers,
            )


if __name__ == "__main__":
    main()
