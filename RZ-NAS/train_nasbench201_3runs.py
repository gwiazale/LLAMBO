"""
Train the best NAS-Bench-201 architecture (from evolution_search_nasbench201.py)
for 3 runs under the benchmark training setting. Reports: Top-1 accuracy, test error,
FLOPs, GPU cost (wall-clock time), and optional regret.

Usage:
  python train_nasbench201_3runs.py --plainnet_struct_txt save_dir/best_structure.txt \\
    --save_dir save_dir/train_3runs --dataset cifar10 --num_classes 10
  # With regret (best possible test acc on benchmark, e.g. from NAS-Bench-201):
  python train_nasbench201_3runs.py ... --best_accuracy 94.36
"""

import os
import sys
import argparse
import logging
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np

import global_utils
from nasbench201_net import build_nasbench201

# CIFAR input size for FLOPs
INPUT_RESOLUTION = 32


def parse_args(argv):
    p = argparse.ArgumentParser(description="Train NAS-Bench-201 arch 3 runs; report top-1, test error, FLOPs, GPU cost, regret.")
    p.add_argument("--plainnet_struct_txt", type=str, required=True,
                   help="Path to best_structure.txt (NAS-Bench-201 arch string).")
    p.add_argument("--save_dir", type=str, required=True,
                   help="Directory to save checkpoints and results.")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs (benchmark uses 200).")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--num_runs", type=int, default=3)
    p.add_argument("--seeds", type=str, default="0,1,2",
                   help="Comma-separated seeds for each run.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--best_accuracy", type=float, default=None,
                   help="Best possible test accuracy (e.g. from NAS-Bench-201) to compute regret = best - mean_top1.")
    return p.parse_args(argv)


def get_cifar_loaders(dataset, batch_size, num_workers=4):
    """CIFAR-10/100 with standard augmentation (pad 4, crop 32, flip)."""
    norm = T.Normalize(
        (0.4914, 0.4822, 0.4465) if dataset == "cifar10" else (0.5071, 0.4867, 0.4408),
        (0.2470, 0.2435, 0.2616) if dataset == "cifar10" else (0.2675, 0.2565, 0.2761),
    )
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        norm,
    ])
    test_tf = T.Compose([T.ToTensor(), norm])

    if dataset == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root=os.environ.get("TORCH_HOME", "./data"), train=True, download=True, transform=train_tf)
        test_ds = torchvision.datasets.CIFAR10(root=os.environ.get("TORCH_HOME", "./data"), train=False, download=True, transform=test_tf)
    else:
        train_ds = torchvision.datasets.CIFAR100(root=os.environ.get("TORCH_HOME", "./data"), train=True, download=True, transform=train_tf)
        test_ds = torchvision.datasets.CIFAR100(root=os.environ.get("TORCH_HOME", "./data"), train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(dim=1).cpu()
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def run_one_training(arch_str, train_loader, test_loader, args, seed, run_dir, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_nasbench201(arch_str, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_test_acc = 0.0
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        test_acc = evaluate(model, test_loader, device)
        best_test_acc = max(best_test_acc, test_acc)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            logging.info("Run seed %s epoch %s train_acc %.2f test_acc %.2f best_test %.2f",
                         seed, epoch + 1, train_acc, test_acc, best_test_acc)
    elapsed = time.perf_counter() - t0
    if device.type == "cuda":
        torch.cuda.synchronize()
    torch.save(model.state_dict(), os.path.join(run_dir, "checkpoint.pt"))
    return best_test_acc, elapsed


def main(args, argv=None):
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    with open(args.plainnet_struct_txt, "r") as f:
        arch_str = f.read().strip()
    if not arch_str:
        logging.error("Empty arch string in %s", args.plainnet_struct_txt)
        return

    global_utils.mkfilepath(args.save_dir)
    train_loader, test_loader = get_cifar_loaders(args.dataset, args.batch_size)
    seeds = [int(s) for s in args.seeds.replace(" ", "").split(",")][: args.num_runs]

    # FLOPs and params (once per architecture)
    model_for_metrics = build_nasbench201(arch_str, num_classes=args.num_classes)
    flops = model_for_metrics.get_FLOPs(INPUT_RESOLUTION)
    n_params = model_for_metrics.get_model_size()
    del model_for_metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_accs = []
    run_times = []
    for i, seed in enumerate(seeds):
        run_dir = os.path.join(args.save_dir, "run%d_seed%d" % (i + 1, seed))
        os.makedirs(run_dir, exist_ok=True)
        logging.info("Starting run %d / %d (seed=%d)", i + 1, len(seeds), seed)
        acc, elapsed = run_one_training(arch_str, train_loader, test_loader, args, seed, run_dir, device)
        test_accs.append(acc)
        run_times.append(elapsed)
        logging.info("Run %d seed %d best top-1 test accuracy: %.2f%%  time: %.1fs", i + 1, seed, acc, elapsed)

    # Aggregates over 3 runs
    mean_acc = float(np.mean(test_accs))
    std_acc = float(np.std(test_accs))
    test_errors = [100.0 - a for a in test_accs]
    mean_error = float(np.mean(test_errors))
    std_error = float(np.std(test_errors))
    total_time = sum(run_times)
    mean_time = total_time / len(run_times) if run_times else 0.0

    regret = None
    if args.best_accuracy is not None:
        regret = float(args.best_accuracy - mean_acc)

    # Build report
    lines = [
        "=" * 60,
        "NAS-Bench-201 3-run training summary",
        "=" * 60,
        "Top-1 test accuracy (best per run):  mean = %.2f%%,  std = %.2f%%  (over %d runs)" % (mean_acc, std_acc, len(test_accs)),
        "Test error (100 - top-1):            mean = %.2f%%,  std = %.2f%%" % (mean_error, std_error),
        "Per-run top-1: %s" % [round(a, 2) for a in test_accs],
        "Per-run test error: %s" % [round(e, 2) for e in test_errors],
        "-" * 60,
        "FLOPs (one forward, %dx%d):         %s" % (INPUT_RESOLUTION, INPUT_RESOLUTION, _format_flops(flops)),
        "Params:                             %s" % _format_params(n_params),
        "-" * 60,
        "GPU cost (wall-clock):              total = %.1f s (%.2f h)  |  per run = %.1f s" % (total_time, total_time / 3600.0, mean_time),
        "Per-run times (s): %s" % [round(t, 1) for t in run_times],
    ]
    if regret is not None:
        lines.append("-" * 60)
        lines.append("Regret (best_acc - mean_top1):   %.2f%%  (best_accuracy=%.2f%%)" % (regret, args.best_accuracy))
    lines.append("=" * 60)

    summary = "\n".join(lines)
    logging.info(summary)

    result_file = os.path.join(args.save_dir, "train_3runs_result.txt")
    with open(result_file, "w") as f:
        f.write(summary + "\n")
    print(summary)


def _format_flops(flops):
    if flops <= 0:
        return "N/A (install thop for FLOPs)"
    if flops >= 1e9:
        return "%.2f GFLOPs" % (flops / 1e9)
    if flops >= 1e6:
        return "%.2f MFLOPs" % (flops / 1e6)
    return "%.0f FLOPs" % flops


def _format_params(n):
    if n >= 1e6:
        return "%.2f M" % (n / 1e6)
    if n >= 1e3:
        return "%.2f K" % (n / 1e3)
    return str(n)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    os.makedirs(args.save_dir, exist_ok=True)
    log_fn = os.path.join(args.save_dir, "train_nasbench201_3runs.log")
    global_utils.create_logging(log_fn)
    main(args)
