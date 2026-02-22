"""
LLAMBO for NAS-Bench-201 neural network optimization.

Tunes: learning rate, weight decay, batch_size, architecture index (discrete),
learning rate scheduler, optimizer choice. Reports test accuracy (score) and FLOPs.

Similar to run_bayesmark.py but for NAS-Bench-201 training. Uses RZ-NAS for
model building and data loaders; runs one short training per BO trial.

Usage (from repo root or exp_bayesmark):
  python exp_bayesmark/run_nasbench201.py --dataset cifar10 --num_seeds 1 --sm_mode discriminative
  python exp_bayesmark/run_nasbench201.py --dataset ImageNet16-120 --data_dir /path/to/ImageNet16-120 --epochs_per_trial 12

  # Warm start from prior (config + arch_index + test_accuracy) results:
  python exp_bayesmark/run_nasbench201.py --dataset cifar10 --sm_mode discriminative --warm_start_json path/to/prior_results.json
"""

import os
import sys
import json
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_RZ_NAS = os.path.join(_REPO_ROOT, "RZ-NAS")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _RZ_NAS not in sys.path:
    sys.path.insert(0, _RZ_NAS)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from llambo.llambo import LLAMBO

# RZ-NAS imports (after path setup)
import global_utils
from nasbench201_net import build_nasbench201, index_to_arch_str
from train_nasbench201_3runs import (
    get_loaders,
    train_one_epoch,
    evaluate,
    DATASET_RESOLUTION,
    _dataset_num_classes,
)

logger = logging.getLogger(__name__)

# Scheduler: 0=cosine, 1=multistep, 2=exponential, 3=none
SCHEDULER_NAMES = ["cosine", "multistep", "exponential", "none"]
# Optimizer: 0=SGD, 1=Adam, 2=AdamW
OPTIMIZER_NAMES = ["SGD", "Adam", "AdamW"]

NUM_ARCHS = 15625  # 5^6 for NAS-Bench-201

# Required config keys for NAS-Bench-201 (must match hp_configurations/nasbench201.json)
NASBENCH201_HP_KEYS = ["lr", "weight_decay", "batch_size", "arch_index", "scheduler", "optimizer"]


def _parse_score(score_val):
    """
    Parse score from either a number or a string 'mean +/- std' / 'mean ± std'.
    Returns (mean_float, std_float_or_None). Optimization uses the mean only.
    """
    if score_val is None:
        return None, None
    if isinstance(score_val, (int, float)):
        return float(score_val), None
    s = str(score_val).strip()
    # Match "0.85 +/- 0.02" or "0.85 ± 0.02" or "0.85 +- 0.02"
    for sep in ("+/-", "±", "+-"):
        if sep in s:
            parts = s.split(sep, 1)
            if len(parts) == 2:
                try:
                    mean = float(parts[0].strip())
                    std = float(parts[1].strip())
                    return mean, std
                except ValueError:
                    break
            break
    # Plain number string
    try:
        return float(s), None
    except ValueError:
        raise ValueError(f"score must be a number or 'mean +/- std'; got: {score_val!r}")


def load_initial_observations(warm_start_input):
    """
    Convert a dictionary of tested hyperparameters + NAS-Bench-201 arch index + test accuracy
    into the list-of-dicts format expected by LLAMBO's initial_observations.

    warm_start_input: either
      - a list of dicts, each with keys lr, weight_decay, batch_size, arch_index, scheduler,
        optimizer, and either 'score' or 'test_accuracy', or
      - a path (str) to a JSON file containing such a list.

    Score can be a number or a string "accuracy +/- standard deviation" (e.g. "0.85 +/- 0.02").
    The mean is used for optimization; the std is stored as score_std when present.

    Example JSON entry:
      {"lr": 0.1, "weight_decay": 1e-5, "batch_size": 128, "arch_index": 42, "scheduler": 0, "optimizer": 0, "test_accuracy": "0.85 +/- 0.02"}

    Returns a list of dicts with keys NASBENCH201_HP_KEYS + 'score' (+ optional 'score_std').
    """
    if isinstance(warm_start_input, str):
        with open(warm_start_input, "r") as f:
            warm_start_input = json.load(f)
    if not isinstance(warm_start_input, list):
        raise TypeError("warm_start_input must be a list of dicts or a path to a JSON file containing that list")
    out = []
    for i, rec in enumerate(warm_start_input):
        if not isinstance(rec, dict):
            raise TypeError(f"Entry {i} must be a dict; got {type(rec)}")
        missing = [k for k in NASBENCH201_HP_KEYS if k not in rec]
        if missing:
            raise KeyError(f"Entry {i} missing hyperparameter keys: {missing}")
        raw_score = rec.get("score", rec.get("test_accuracy"))
        if raw_score is None:
            raise KeyError(f"Entry {i} must contain 'score' or 'test_accuracy'")
        mean_score, std_score = _parse_score(raw_score)
        row = {**{k: rec[k] for k in NASBENCH201_HP_KEYS}, "score": mean_score}
        if std_score is not None:
            row["score_std"] = std_score
        out.append(row)
    return out


def setup_logging(log_name):
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(log_name, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


def _sample_config(constraints, rng):
    """Sample one random config from hyperparameter_constraints (numeric bounds only)."""
    config = {}
    for name, (dtype, transform, (lo, hi)) in constraints.items():
        u = rng.uniform(0, 1)
        if transform == "log":
            val = 10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))
        else:
            val = lo + u * (hi - lo)
        if dtype == "int":
            val = int(np.clip(round(val), lo, hi))
        config[name] = val
    return config


def _get_optimizer(optimizer_index, model, lr, weight_decay):
    if optimizer_index == 0:  # SGD
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    if optimizer_index == 1:  # Adam
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_index == 2:  # AdamW
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError("optimizer_index must be 0, 1, or 2")


def _get_scheduler(scheduler_index, optimizer, epochs):
    if scheduler_index == 0:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_index == 1:
        # MultiStepLR: reduce at 30%, 60%, 80% of epochs
        milestones = [int(epochs * 0.3), int(epochs * 0.6), int(epochs * 0.8)]
        return optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.2
        )
    if scheduler_index == 2:
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # 3 = none
    return None


def run_one_training_trial(
    arch_str,
    train_loader,
    test_loader,
    num_classes,
    lr,
    weight_decay,
    optimizer_index,
    scheduler_index,
    epochs,
    seed,
    device,
):
    """Run a single training run; return best test accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_nasbench201(arch_str, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _get_optimizer(optimizer_index, model, lr, weight_decay)
    scheduler = _get_scheduler(scheduler_index, optimizer, epochs)

    best_test_acc = 0.0
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        if scheduler is not None:
            scheduler.step()
        test_acc = evaluate(model, test_loader, device)
        best_test_acc = max(best_test_acc, test_acc)
    return best_test_acc


class Nasbench201ExpRunner:
    def __init__(self, task_context, dataset_name, data_dir, seed, epochs_per_trial, gpu, num_runs_per_trial=1):
        self.seed = seed
        self.task_context = task_context
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.epochs_per_trial = epochs_per_trial
        self.gpu = gpu
        self.num_runs_per_trial = num_runs_per_trial
        self.hyperparameter_constraints = task_context["hyperparameter_constraints"]
        self.num_classes = _dataset_num_classes(dataset_name)
        self.device = torch.device(
            "cuda:%d" % gpu if torch.cuda.is_available() and gpu >= 0 else "cpu"
        )
        self.rng = np.random.default_rng(seed)

    def generate_initialization(self, n_samples):
        """Generate random initial configurations for BO."""
        configs = [_sample_config(self.hyperparameter_constraints, self.rng) for _ in range(n_samples)]
        assert len(configs) == n_samples
        return configs

    def evaluate_point(self, candidate_config):
        """
        Evaluate one config: train NAS-Bench-201 with given hyperparameters;
        return (config, fvals) with fvals containing 'score' (test accuracy) and 'flops'.
        """
        # Copy so we don't mutate the original
        config = dict(candidate_config)
        constraints = self.hyperparameter_constraints

        # Cast types
        for name, value in config.items():
            if constraints[name][0] == "int":
                lo, hi = constraints[name][2]
                config[name] = int(np.clip(round(value), lo, hi))

        lr = float(config["lr"])
        weight_decay = float(config["weight_decay"])
        batch_size = int(config["batch_size"])
        arch_index = int(config["arch_index"])
        scheduler_index = int(config["scheduler"])
        optimizer_index = int(config["optimizer"])

        arch_index = np.clip(arch_index, 0, NUM_ARCHS - 1)
        arch_str = index_to_arch_str(arch_index)

        # Data loaders for this batch_size and dataset
        loader_args = SimpleNamespace(
            dataset=self.dataset_name,
            batch_size=batch_size,
            data_dir=self.data_dir,
        )
        train_loader, test_loader, num_classes_override = get_loaders(loader_args, num_workers=0)
        num_classes = num_classes_override if num_classes_override is not None else self.num_classes

        # FLOPs for this architecture (once)
        resolution = DATASET_RESOLUTION.get(self.dataset_name, 32)
        model_for_flops = build_nasbench201(arch_str, num_classes=num_classes)
        flops = model_for_flops.get_FLOPs(resolution)
        del model_for_flops
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Train one or more runs and average score
        np.random.seed(self.seed)
        random.seed(self.seed)
        accs = []
        for r in range(self.num_runs_per_trial):
            run_seed = self.seed + r * 1000
            acc = run_one_training_trial(
                arch_str=arch_str,
                train_loader=train_loader,
                test_loader=test_loader,
                num_classes=num_classes,
                lr=lr,
                weight_decay=weight_decay,
                optimizer_index=optimizer_index,
                scheduler_index=scheduler_index,
                epochs=self.epochs_per_trial,
                seed=run_seed,
                device=self.device,
            )
            accs.append(acc)
        score = float(np.mean(accs))

        fvals = {"score": score, "flops": float(flops)}
        if self.num_runs_per_trial > 1:
            fvals["score_std"] = float(np.std(accs))
        return config, fvals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLAMBO for NAS-Bench-201: tune lr, weight_decay, batch_size, arch_index, scheduler, optimizer."
    )
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "ImageNet16-120"],
                        help="Dataset for training.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Root dir for ImageNet16-120 (optional).")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--engine", type=str,
                        default=os.environ.get("OPENAI_API_ENGINE", "gpt-4o"),
                        help="Chat model/engine.")
    parser.add_argument("--sm_mode", type=str, required=True,
                        choices=["discriminative", "generative"],
                        help="Surrogate model mode.")
    parser.add_argument("--epochs_per_trial", type=int, default=12,
                        help="Training epochs per BO trial (use 200 for full benchmark).")
    parser.add_argument("--num_runs_per_trial", type=int, default=1,
                        help="Number of training runs per config (1 = fast BO).")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_trials", type=int, default=25)
    parser.add_argument("--n_initial_samples", type=int, default=5)
    parser.add_argument("--warm_start_json", type=str, default=None,
                        help="Path to JSON file: list of dicts with keys lr, weight_decay, batch_size, arch_index, scheduler, optimizer, and test_accuracy (or score). Seeds BO with these prior results; no random init.")
    args = parser.parse_args()

    if args.sm_mode == "generative":
        top_pct = 0.25
    else:
        top_pct = None

    # Task context for NAS-Bench-201 (classification, maximize accuracy)
    hp_path = os.path.join(_REPO_ROOT, "hp_configurations", "nasbench201.json")
    with open(hp_path, "r") as f:
        hp_config = json.load(f)
    task_context = {
        "model": "NASBench201",
        "task": "classification",
        "metric": "accuracy",
        "lower_is_better": False,
        "hyperparameter_constraints": hp_config["NASBench201"],
        "num_classes": _dataset_num_classes(args.dataset),
        "dataset_type": "image",
        "dataset_name": args.dataset,
    }

    save_res_dir = os.path.join(
        _SCRIPT_DIR, "results_nasbench201", args.sm_mode, args.dataset
    )
    os.makedirs(save_res_dir, exist_ok=True)
    logging_fpath = os.path.join(
        _SCRIPT_DIR, "logs_nasbench201", args.sm_mode, f"{args.dataset}.log"
    )
    os.makedirs(os.path.dirname(logging_fpath), exist_ok=True)
    setup_logging(logging_fpath)

    tot_llm_cost = 0
    for seed in range(args.num_seeds):
        logger.info("=" * 120)
        logger.info(
            "LLAMBO (%s) NAS-Bench-201 on %s seed %d/%d (epochs_per_trial=%d)",
            args.sm_mode, args.dataset, seed + 1, args.num_seeds, args.epochs_per_trial,
        )
        logger.info("Task context: %s", task_context)

        benchmark = Nasbench201ExpRunner(
            task_context=task_context,
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            seed=seed,
            epochs_per_trial=args.epochs_per_trial,
            gpu=args.gpu,
            num_runs_per_trial=args.num_runs_per_trial,
        )

        initial_observations = None
        if args.warm_start_json:
            initial_observations = load_initial_observations(args.warm_start_json)
            logger.info("Warm start: loaded %d prior observations from %s", len(initial_observations), args.warm_start_json)

        llambo = LLAMBO(
            task_context,
            args.sm_mode,
            n_candidates=10,
            n_templates=2,
            n_gens=10,
            alpha=0.1,
            n_initial_samples=args.n_initial_samples,
            n_trials=args.n_trials,
            init_f=benchmark.generate_initialization,
            bbox_eval_f=benchmark.evaluate_point,
            chat_engine=args.engine,
            top_pct=top_pct,
            initial_observations=initial_observations,
        )
        llambo.seed = seed
        configs, fvals = llambo.optimize()

        logger.info("LLAMBO query cost: %s", sum(llambo.llm_query_cost))
        logger.info("LLAMBO query time: %s", sum(llambo.llm_query_time))
        tot_llm_cost += sum(llambo.llm_query_cost)

        search_history = pd.concat([configs, fvals], axis=1)
        out_csv = os.path.join(save_res_dir, f"{seed}.csv")
        search_history.to_csv(out_csv, index=False)
        logger.info("Saved results to %s", out_csv)
        logger.info("%s", search_history)

        search_info = {
            "llm_query_cost_breakdown": llambo.llm_query_cost,
            "llm_query_time_breakdown": llambo.llm_query_time,
            "llm_query_cost": sum(llambo.llm_query_cost),
            "llm_query_time": sum(llambo.llm_query_time),
        }
        with open(os.path.join(save_res_dir, f"{seed}_search_info.json"), "w") as f:
            json.dump(search_info, f)

    logger.info("=" * 120)
    logger.info("NAS-Bench-201 BO complete. Total LLM cost: %s", tot_llm_cost)
