"""
Train the best NAS-Bench-201 architecture (from evolution_search_nasbench201.py)
for 3 runs under the benchmark training setting. Reports: Top-1 accuracy, test error,
FLOPs, GPU cost (wall-clock time), and optional regret.

Usage:
  python train_nasbench201_3runs.py --plainnet_struct_txt save_dir/best_structure.txt \\
    --save_dir save_dir/train_3runs --dataset cifar10 --num_classes 10
  # ImageNet-16-120 (120 classes, 16x16):
  python train_nasbench201_3runs.py ... --dataset ImageNet16-120 --data_dir /path/to/ImageNet16-120
  # With regret (best possible test acc on benchmark, e.g. from NAS-Bench-201):
  python train_nasbench201_3runs.py ... --best_accuracy 94.36
"""

import os
import sys
import argparse
import logging
import time
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
import numpy as np

import global_utils
from nasbench201_net import build_nasbench201

# Input resolution per dataset (for FLOPs)
DATASET_RESOLUTION = {"cifar10": 32, "cifar100": 32, "ImageNet16-120": 16}


def parse_args(argv):
    p = argparse.ArgumentParser(description="Train NAS-Bench-201 arch 3 runs; report top-1, test error, FLOPs, GPU cost, regret.")
    p.add_argument("--plainnet_struct_txt", type=str, required=True,
                   help="Path to best_structure.txt (NAS-Bench-201 arch string).")
    p.add_argument("--save_dir", type=str, required=True,
                   help="Directory to save checkpoints and results.")
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=["cifar10", "cifar100", "ImageNet16-120"],
                   help="cifar10/cifar100 auto-download; ImageNet16-120 requires --data_dir.")
    p.add_argument("--num_classes", type=int, default=None,
                   help="Auto from dataset: 10 (cifar10), 100 (cifar100), 120 (ImageNet16-120).")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Root for ImageNet16-120: either (1) CIFAR-format dir with train_data_batch_1, val_data, or (2) ImageFolder with train/ and val/. Default: ./data/ImageNet16 or TORCH_HOME/ImageNet16-120.")
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


def _dataset_num_classes(dataset):
    if dataset == "cifar10":
        return 10
    if dataset == "cifar100":
        return 100
    if dataset == "ImageNet16-120":
        return 120
    return 10


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


class ImageNet16CIFAR(Dataset):
    """ImageNet-16-120 in CIFAR-style pickle format: train_data_batch_1, train_data_batch_2, ..., val_data.
    Each batch: dict with 'data' (N x 768 uint8 for 16x16x3) and 'labels' (list). Keys may be bytes in pickle."""
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        if train:
            batch_files = []
            for i in range(1, 20):
                p = os.path.join(data_dir, "train_data_batch_%d" % i)
                if os.path.isfile(p):
                    batch_files.append(p)
            if not batch_files:
                raise FileNotFoundError("No train_data_batch_* found in %s" % data_dir)
            for p in sorted(batch_files):
                with open(p, "rb") as f:
                    d = pickle.load(f, encoding="bytes")
                data_key = b"data" if b"data" in d else "data"
                labels_key = b"labels" if b"labels" in d else "labels"
                self.data.append(d[data_key])
                self.labels.extend(d[labels_key])
        else:
            for name in ("val_data", "test_data"):
                p = os.path.join(data_dir, name)
                if os.path.isfile(p):
                    with open(p, "rb") as f:
                        d = pickle.load(f, encoding="bytes")
                    data_key = b"data" if b"data" in d else "data"
                    labels_key = b"labels" if b"labels" in d else "labels"
                    self.data.append(d[data_key])
                    self.labels.extend(d[labels_key])
                    break
            if not self.data:
                raise FileNotFoundError("Neither val_data nor test_data found in %s" % data_dir)
        self.data = np.vstack(self.data)  # (N, 768)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.data[i].reshape(3, 16, 16).transpose(1, 2, 0)  # (16, 16, 3) HWC
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[i]


def _has_cifar_format_imagenet16(data_dir):
    """True if data_dir contains CIFAR-format batch files (train_data_batch_1 or train_data_batch_2, etc.)."""
    if not data_dir or not os.path.isdir(data_dir):
        return False
    for name in ("train_data_batch_1", "train_data_batch_2"):
        if os.path.isfile(os.path.join(data_dir, name)):
            return True
    return False


def get_imagenet16_120_loaders(data_dir, batch_size, num_workers=4):
    """ImageNet-16-120: 120 classes, 16x16 images.
    Supports two layouts:
    (1) CIFAR-format: data_dir contains train_data_batch_1, train_data_batch_2, ..., val_data (or test_data).
    (2) ImageFolder: data_dir contains train/ and val/ (or test/) with class subdirs."""
    norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_tf = T.Compose([
        T.ToPILImage(),
        T.RandomCrop(16, padding=2),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        norm,
    ])
    test_tf = T.Compose([T.ToPILImage(), T.ToTensor(), norm])

    if _has_cifar_format_imagenet16(data_dir):
        train_ds = ImageNet16CIFAR(data_dir, train=True, transform=train_tf)
        test_ds = ImageNet16CIFAR(data_dir, train=False, transform=test_tf)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader

    # ImageFolder layout
    train_root = os.path.join(data_dir, "train")
    val_root = os.path.join(data_dir, "val")
    if not os.path.isdir(val_root):
        val_root = os.path.join(data_dir, "test")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(
            "ImageNet16-120 not found in %s: expected either (1) CIFAR-format files train_data_batch_1, val_data, "
            "or (2) ImageFolder layout with train/ and val/ subdirs. Set --data_dir to your dataset root."
            % data_dir
        )
    if not os.path.isdir(val_root):
        raise FileNotFoundError("ImageNet16-120 val/test dir not found: %s" % val_root)
    train_tf_img = T.Compose([
        T.RandomCrop(16, padding=2),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        norm,
    ])
    test_tf_img = T.Compose([T.ToTensor(), norm])
    train_ds = torchvision.datasets.ImageFolder(train_root, transform=train_tf_img)
    test_ds = torchvision.datasets.ImageFolder(val_root, transform=test_tf_img)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def _default_imagenet16_data_dir():
    """Default root for ImageNet16-120: try RZ-NAS/data/ImageNet16 (CIFAR-format), then TORCH_HOME/ImageNet16-120."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, "data", "ImageNet16")
    if os.path.isdir(candidate):
        return candidate
    candidate = os.path.join(os.environ.get("TORCH_HOME", "./data"), "ImageNet16")
    if os.path.isdir(candidate):
        return candidate
    return os.path.join(os.environ.get("TORCH_HOME", "./data"), "ImageNet16-120")


def get_loaders(args, num_workers=4):
    """Return (train_loader, test_loader) for cifar10, cifar100, or ImageNet16-120."""
    if args.dataset == "ImageNet16-120":
        data_dir = args.data_dir or _default_imagenet16_data_dir()
        return get_imagenet16_120_loaders(data_dir, args.batch_size, num_workers=num_workers)
    return get_cifar_loaders(args.dataset, args.batch_size, num_workers=num_workers)


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

    if args.num_classes is None:
        args.num_classes = _dataset_num_classes(args.dataset)
    global_utils.mkfilepath(args.save_dir)
    train_loader, test_loader = get_loaders(args)
    seeds = [int(s) for s in args.seeds.replace(" ", "").split(",")][: args.num_runs]

    input_resolution = DATASET_RESOLUTION.get(args.dataset, 32)
    # FLOPs and params (once per architecture)
    model_for_metrics = build_nasbench201(arch_str, num_classes=args.num_classes)
    flops = model_for_metrics.get_FLOPs(input_resolution)
    n_params = model_for_metrics.get_model_size()
    del model_for_metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logging.info("Will report at end: top-1 mean±std, test error mean±std, GPU cost (wall-clock), regret (if --best_accuracy given).")
    test_accs = []
    run_times = []
    for i, seed in enumerate(seeds):
        run_dir = os.path.join(args.save_dir, "run%d_seed%d" % (i + 1, seed))
        os.makedirs(run_dir, exist_ok=True)
        logging.info("Starting run %d / %d (seed=%d)", i + 1, len(seeds), seed)
        acc, elapsed = run_one_training(arch_str, train_loader, test_loader, args, seed, run_dir, device)
        test_accs.append(acc)
        run_times.append(elapsed)
        logging.info("Run %d seed %d best top-1 test accuracy: %.2f%%  test_error: %.2f%%  time: %.1fs", i + 1, seed, acc, 100.0 - acc, elapsed)
        # Intermediate summary after each run (so you see mean±std, test error, GPU cost even if stopped early)
        n = len(test_accs)
        run_errs = [100.0 - a for a in test_accs]
        m_acc, s_acc = float(np.mean(test_accs)), float(np.std(test_accs)) if n > 1 else (float(np.mean(test_accs)), 0.0)
        m_err, s_err = float(np.mean(run_errs)), float(np.std(run_errs)) if n > 1 else (float(np.mean(run_errs)), 0.0)
        total_time_so_far = sum(run_times)
        logging.info("[After run %d] top-1: mean=%.2f%%, std=%.2f%%  |  test_error: mean=%.2f%%, std=%.2f%%  |  GPU_cost_total=%.1fs", n, m_acc, s_acc, m_err, s_err, total_time_so_far)

    # Aggregates over all runs
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
        "FLOPs (one forward, %dx%d):         %s" % (input_resolution, input_resolution, _format_flops(flops)),
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
    print(summary, flush=True)
    logging.info("Summary also written to %s", result_file)


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
