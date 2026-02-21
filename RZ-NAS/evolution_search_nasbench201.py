"""
Evolution search over NAS-Bench-201 search space.

Two modes (paper setting: use proxy, then train best 3 runs):

1) Proxy as fitness (recommended): --zero_shot_score Zen|TE-NAS|Syncflow|GradNorm|NASWOT|Flops|Params|Random
   - Search uses zero-shot proxy score; no benchmark file required.
   - Output: best_structure.txt (NAS-Bench-201 arch string). Then train that arch 3 runs
     under the benchmark training setting and report accuracy (use train_nasbench201_3runs.py).

2) Precomputed as fitness: omit --zero_shot_score, pass --benchmark_path
   - Fitness = validation/test accuracy from the benchmark .pth (no training during search).

Requires for proxy mode: PyTorch, RZ-NAS ZeroShotProxy and nasbench201_net.
Optional: pip install nas-bench-201 (and benchmark file) for precomputed mode or for
  sampling via API; otherwise archs are generated via index <-> string conversion.
"""

import os
import sys
import argparse
import random
import logging
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import global_utils

from nasbench201_net import (
    build_nasbench201,
    mutate_arch_string_simple,
    index_to_arch_str,
    arch_str_to_index,
    arch_str_to_ops,
    OPS,
)

try:
    from nas_201_api import NASBench201API
except ImportError:
    NASBench201API = None

# Proxy imports (same as evolution_search.py)
try:
    from ZeroShotProxy import (
        compute_zen_score,
        compute_te_nas_score,
        compute_syncflow_score,
        compute_gradnorm_score,
        compute_NASWOT_score,
    )
except ImportError:
    compute_zen_score = compute_te_nas_score = compute_syncflow_score = None
    compute_gradnorm_score = compute_NASWOT_score = None


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser(
        description="Evolution search on NAS-Bench-201 (proxy or precomputed fitness)."
    )
    parser.add_argument(
        "--zero_shot_score",
        type=str,
        default=None,
        help="Zero-shot proxy for fitness: Zen, TE-NAS, Syncflow, GradNorm, NASWOT, Flops, Params, Random. "
        "If set, fitness = proxy score (no benchmark file needed).",
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default=None,
        help="Path to NAS-Bench-201 .pth (only for precomputed fitness when --zero_shot_score is not set).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Dataset for precomputed fitness (ignored in proxy mode).",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="200",
        choices=["12", "200"],
        help="Epochs in benchmark for precomputed (ignored in proxy mode).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="valid",
        choices=["valid", "test"],
        help="Metric for precomputed fitness (ignored in proxy mode).",
    )
    parser.add_argument(
        "--evolution_max_iter",
        type=int,
        default=50000,
        help="Max iterations of evolution.",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=256,
        help="Population size.",
    )
    parser.add_argument(
        "--mutate_random_prob",
        type=float,
        default=0.2,
        help="Probability of replacing with a random architecture.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        required=True,
        help="Output directory for best_structure.txt and logs.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id for proxy computation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for proxy forward.",
    )
    parser.add_argument(
        "--input_image_size",
        type=int,
        default=32,
        help="Input resolution (32 for CIFAR).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of classes (10 for CIFAR-10, 100 for CIFAR-100).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1e-2,
        help="Mixup gamma for Zen-NAS.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_known_args(argv)[0]


# ---------- Proxy fitness: build NAS-Bench-201 net and run zero-shot score ----------
def compute_nas_score_nasbench201(arch_str, gpu, args):
    """Compute zero-shot proxy score for a NAS-Bench-201 arch string."""
    model = build_nasbench201(arch_str, num_classes=args.num_classes)
    if gpu is not None and torch.cuda.is_available():
        model = model.cuda(gpu)
    try:
        if args.zero_shot_score == "Zen":
            info = compute_zen_score.compute_nas_score(
                model=model,
                gpu=gpu,
                resolution=args.input_image_size,
                mixup_gamma=args.gamma,
                batch_size=args.batch_size,
                repeat=1,
            )
            score = info["avg_nas_score"]
        elif args.zero_shot_score == "TE-NAS":
            score = compute_te_nas_score.compute_NTK_score(
                model=model,
                gpu=gpu,
                resolution=args.input_image_size,
                batch_size=args.batch_size,
            )
        elif args.zero_shot_score == "Syncflow":
            score = compute_syncflow_score.do_compute_nas_score(
                model=model,
                gpu=gpu,
                resolution=args.input_image_size,
                batch_size=args.batch_size,
            )
        elif args.zero_shot_score == "GradNorm":
            score = compute_gradnorm_score.compute_nas_score(
                model=model,
                gpu=gpu,
                resolution=args.input_image_size,
                batch_size=args.batch_size,
            )
        elif args.zero_shot_score == "NASWOT":
            score = compute_NASWOT_score.compute_nas_score(
                gpu=gpu,
                model=model,
                resolution=args.input_image_size,
                batch_size=args.batch_size,
            )
        elif args.zero_shot_score == "Flops":
            score = model.get_FLOPs(args.input_image_size)
        elif args.zero_shot_score == "Params":
            score = model.get_model_size()
        elif args.zero_shot_score == "Random":
            score = np.random.randn()
        else:
            logging.warning("Unknown zero_shot_score: %s", args.zero_shot_score)
            score = -9999.0
    except Exception as err:
        logging.debug("Proxy failed for arch: %s", err)
        score = -9999.0
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score


# ---------- Precomputed fitness (benchmark lookup) ----------
def _get_accuracy_from_info(info):
    if not isinstance(info, dict):
        return None
    for key in ("valid-accuracy", "valid_accuracy", "val_acc", "valid_acc", "x-valid-accuracy"):
        if key in info and info[key] is not None:
            return float(info[key])
    for key, v in info.items():
        if "valid" in key.lower() and "acc" in key.lower() and v is not None:
            return float(v)
    if "test-accuracy" in info:
        return float(info["test-accuracy"])
    if "test_accuracy" in info:
        return float(info["test_accuracy"])
    return None


def get_fitness_precomputed(api, arch_index, dataset, epochs, metric="valid", is_random=True):
    try:
        info = api.get_more_info(arch_index, dataset, None, hp=str(epochs), is_random=is_random)
    except Exception as e:
        logging.warning("get_more_info failed for index %s: %s", arch_index, e)
        return -1.0
    acc = _get_accuracy_from_info(info)
    if acc is not None:
        return acc
    try:
        meta = api.query_meta_info_by_index(arch_index)
        use_12 = str(epochs) == "12"
        split = "x-valid" if metric == "valid" else "x-test"
        res = meta.get_metrics(dataset, split, None, use_12)
        if isinstance(res, dict) and "accuracy" in res:
            return float(res["accuracy"])
        if isinstance(res, (list, tuple)) and len(res) >= 2:
            return float(res[1])
    except Exception:
        pass
    return -1.0


# ---------- Sampling: with or without API ----------
NUM_ARCHS = 5 ** 6  # 15625


def get_random_arch_str(api=None):
    if api is not None:
        idx = random.randint(0, NUM_ARCHS - 1)
        return api[idx]
    return index_to_arch_str(random.randint(0, NUM_ARCHS - 1))


def resolve_arch_str(arch_str, api=None):
    """Return (arch_str, arch_index or None). If API given, resolve index."""
    if api is not None:
        try:
            idx = api.query_index_by_arch(arch_str)
            return arch_str, idx
        except Exception:
            pass
    try:
        idx = arch_str_to_index(arch_str)
        return arch_str, idx
    except Exception:
        return arch_str, None


def main(args):
    use_proxy = args.zero_shot_score is not None and args.zero_shot_score.strip() != ""

    if use_proxy:
        # Proxy mode: no benchmark file required
        api = None
        if NASBench201API is not None and args.benchmark_path and os.path.isfile(args.benchmark_path):
            api = NASBench201API(args.benchmark_path, verbose=False)
    else:
        # Precomputed mode: need benchmark file
        if NASBench201API is None:
            logging.error("Precomputed mode requires: pip install nas-bench-201")
            return None
        benchmark_path = args.benchmark_path or os.path.join(
            os.environ.get("TORCH_HOME", os.path.expanduser("~/.torch")),
            "NAS-Bench-201-v1_1-096897.pth",
        )
        if not os.path.isfile(benchmark_path):
            logging.error("Benchmark file not found: %s", benchmark_path)
            return None
        api = NASBench201API(benchmark_path, verbose=False)

    random.seed(args.seed)
    if args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.backends.cudnn.benchmark = True
    else:
        args.gpu = None

    best_structure_txt = os.path.join(args.save_dir, "best_structure.txt")
    if os.path.isfile(best_structure_txt):
        logging.info("Output already exists, skip: %s", best_structure_txt)
        return None

    popu_structure_list = []
    popu_index_list = []
    popu_score_list = []

    start_timer = time.time()
    num_archs = len(api) if api is not None else NUM_ARCHS

    for loop_count in range(args.evolution_max_iter):
        while len(popu_structure_list) > args.population_size:
            min_score = min(popu_score_list)
            idx_remove = popu_score_list.index(min_score)
            popu_score_list.pop(idx_remove)
            popu_structure_list.pop(idx_remove)
            popu_index_list.pop(idx_remove)

        if loop_count >= 1 and loop_count % 1000 == 0:
            max_s = max(popu_score_list) if popu_score_list else 0
            min_s = min(popu_score_list) if popu_score_list else 0
            elapsed = time.time() - start_timer
            logging.info(
                "loop_count=%s/%s, max_score=%.4f, min_score=%.4f, time=%.2fh",
                loop_count, args.evolution_max_iter, max_s, min_s, elapsed / 3600,
            )

        if len(popu_structure_list) < 2:
            arch_str = get_random_arch_str(api)
        else:
            if random.random() < args.mutate_random_prob:
                arch_str = get_random_arch_str(api)
            else:
                parent_str = random.choice(popu_structure_list)
                arch_str = mutate_arch_string_simple(parent_str)
                if arch_str is None:
                    arch_str = get_random_arch_str(api)
        arch_str, arch_index = resolve_arch_str(arch_str, api)
        if arch_index is None:
            try:
                arch_index = arch_str_to_index(arch_str)
            except Exception:
                arch_index = -1

        if use_proxy:
            score = compute_nas_score_nasbench201(arch_str, args.gpu, args)
        else:
            score = get_fitness_precomputed(
                api, arch_index, args.dataset, args.epochs,
                metric=args.metric, is_random=True,
            )

        popu_structure_list.append(arch_str)
        popu_index_list.append(arch_index if arch_index is not None else -1)
        popu_score_list.append(score)

    return popu_structure_list, popu_index_list, popu_score_list


if __name__ == "__main__":
    args = parse_cmd_options(sys.argv)
    log_fn = os.path.join(args.save_dir, "evolution_search_nasbench201.log")
    global_utils.mkfilepath(log_fn)
    global_utils.create_logging(log_fn)

    info = main(args)
    if info is None:
        sys.exit(0)

    popu_structure_list, popu_index_list, popu_score_list = info

    best_idx = max(range(len(popu_score_list)), key=lambda i: popu_score_list[i])
    best_structure_str = popu_structure_list[best_idx]
    best_arch_index = popu_index_list[best_idx]
    best_score = popu_score_list[best_idx]

    best_structure_txt = os.path.join(args.save_dir, "best_structure.txt")
    global_utils.mkfilepath(best_structure_txt)
    with open(best_structure_txt, "w") as f:
        f.write(best_structure_str)

    score_type = args.zero_shot_score if args.zero_shot_score else "precomputed"
    logging.info("Best arch index=%s, %s score=%.4f", best_arch_index, score_type, best_score)

    top_k = 50
    indexed = list(zip(popu_score_list, popu_index_list, popu_structure_list))
    indexed.sort(key=lambda x: x[0], reverse=True)
    top50 = indexed[:top_k]
    top50_txt = os.path.join(args.save_dir, "top50_structures.txt")
    with open(top50_txt, "w") as f:
        f.write("# Top {} structures: rank | score | arch_index | structure\n".format(top_k))
        f.write("# rank\tscore\tarch_index\tstructure\n")
        for rank, (score, arch_index, structure) in enumerate(top50, start=1):
            f.write("{}\t{}\t{}\t{}\n".format(rank, score, arch_index, structure))
