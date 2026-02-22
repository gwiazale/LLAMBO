"""
Evolution search over NAS-Bench-201 search space.

RZ-NAS innovation: LLM proposes new candidates (with --use_llm). Population is initialized
with random sampling; then each step can use the LLM to mutate a parent arch for a better
proxy score, or fall back to random/mutation.

Two modes (paper setting: use proxy, then train best 3 runs):

1) Proxy as fitness (recommended): --zero_shot_score Zen|TE-NAS|... and optionally --use_llm
   - Search uses zero-shot proxy; LLM can propose mutated cells when --use_llm.
   - Output: best_structure.txt. Then train with train_nasbench201_3runs.py.

2) Precomputed as fitness: omit --zero_shot_score, pass --benchmark_path
   - Fitness = benchmark lookup (no LLM).
"""

import os
import sys
import argparse
import random
import logging
import time
import re

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
        default=1500,
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
        default=None,
        help="Input resolution (auto from --dataset: 32 for CIFAR, 16 for ImageNet16-120).",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes (auto from --dataset: 10/100/120 for cifar10/cifar100/ImageNet16-120).",
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
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--search_mode",
        type=str,
        default="llm",
        choices=["random", "llm", "mutation"],
        help="How to propose new candidates: 'random' = pure random sampling; "
        "'llm' = LLM proposes mutations (with mutation/random fallback); "
        "'mutation' = evolution by mutating one op or random (no LLM).",
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Deprecated: use --search_mode mutation instead. Disable LLM; use only random + mutation.",
    )
    parser.add_argument(
        "--llm_prob",
        type=float,
        default=0.5,
        help="Probability of using LLM to propose next candidate (only when --search_mode llm).",
    )
    parser.add_argument(
        "--llm_template",
        type=str,
        default=None,
        help="Path to NAS-Bench-201 LLM prompt template (default: prompt/template_nasbench201.txt).",
    )
    parser.add_argument(
        "--random_init_only",
        action="store_true",
        help="No evolution: build initial population (random sample), score with proxy/precomputed, return best. Same as --search_mode random --evolution_max_iter 0.",
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
    # Ensure finite score so min/max/sort never see -inf or nan
    if not np.isfinite(score):
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


def _is_valid_nasbench201_structure(arch_str):
    """Return True if arch_str is a valid NAS-Bench-201 cell (6 edges, valid ops)."""
    if not arch_str or not isinstance(arch_str, str):
        return False
    s = arch_str.strip()
    for prefix in ("```", "Here is", "The structure is", "Output:", "Architecture:"):
        if s.lower().startswith(prefix.lower()):
            idx = s.find("|")
            if idx >= 0:
                s = s[idx:]
            break
    if "```" in s:
        s = re.sub(r"```\w*\n?", "", s).strip()
        if "|" in s:
            s = s[s.find("|"):]
    try:
        ops = arch_str_to_ops(s)
        return len(ops) == 6 and all(op in OPS for op in ops)
    except Exception:
        return False


def _extract_arch_str_from_llm_response(text):
    """Extract a single NAS-Bench-201 arch string from LLM response (JSON with "arch" key or raw string)."""
    if not text:
        return None
    # Try JSON format first (e.g. {"arch":"|...|","type":"Zen","score":1.2})
    try:
        import json
        # Strip code block if present
        s = text.strip().strip("`").strip()
        if s.startswith("json"):
            s = s[4:].strip()
        obj = json.loads(s)
        if isinstance(obj, dict) and "arch" in obj:
            return obj["arch"].strip()
    except Exception:
        pass
    # Look for pattern |op~0|+|...
    match = re.search(r"\|[a-z_0-9~]+\|\+\|[a-z_0-9~]+\|[a-z_0-9~]+\|\+\|[a-z_0-9~]+\|[a-z_0-9~]+\|[a-z_0-9~]+\|", text)
    if match:
        return match.group(0)
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip().strip("`").strip()
        if line.startswith("|") and "~" in line and "+" in line:
            return line
    return None


def generate_by_llm_nasbench201(arch_str, proxy_name, score, template_path=None):
    """Ask LLM to propose a mutated NAS-Bench-201 cell. Returns new arch string or None on failure."""
    try:
        from dotenv import load_dotenv
        import openai
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            logging.warning("OPENAI_API_KEY not set; LLM disabled.")
            return None
    except ImportError:
        logging.warning("openai or dotenv not installed; LLM disabled.")
        return None

    if template_path is None:
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt", "template_nasbench201.txt")
    if not os.path.isfile(template_path):
        logging.warning("LLM template not found: %s", template_path)
        return None

    with open(template_path, "r") as f:
        prompt = f.read()
    prompt = prompt.replace("{{ARCH_STR}}", arch_str)
    prompt = prompt.replace("{{PROXY_NAME}}", str(proxy_name))
    prompt = prompt.replace("{{SCORE}}", str(score))

    for attempt in range(3):
        try:
            kwargs = dict(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            response = openai.ChatCompletion.create(**kwargs)
            raw = response["choices"][0]["message"]["content"].strip()
            new_str = _extract_arch_str_from_llm_response(raw)
            if new_str and _is_valid_nasbench201_structure(new_str):
                return new_str
            break
        except Exception as e:
            if "server_error" in str(e).lower() or "server had an error" in str(e).lower():
                if attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                    continue
            logging.warning("LLM call failed (attempt %d): %s. Using mutation fallback.", attempt + 1, e)
            break
    return None


def get_new_structure_str_with_llm(parent_arch_str, parent_score, proxy_name, api, args):
    """Try LLM first to get a new candidate; fall back to mutation or random."""
    template_path = getattr(args, "llm_template", None)
    new_str = generate_by_llm_nasbench201(parent_arch_str, proxy_name, parent_score, template_path)
    if new_str is not None:
        return new_str
    # Fallback: mutate one op
    new_str = mutate_arch_string_simple(parent_arch_str)
    if new_str is not None:
        return new_str
    return get_random_arch_str(api)


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


def _set_dataset_defaults(args):
    """Set num_classes and input_image_size from --dataset when not explicitly provided."""
    dataset = getattr(args, "dataset", "cifar10")
    if getattr(args, "num_classes", None) is None:
        if dataset == "cifar10":
            args.num_classes = 10
        elif dataset == "cifar100":
            args.num_classes = 100
        elif dataset == "ImageNet16-120":
            args.num_classes = 120
        else:
            args.num_classes = 10
    if getattr(args, "input_image_size", None) is None:
        if dataset == "ImageNet16-120":
            args.input_image_size = 16
        else:
            args.input_image_size = 32


def main(args):
    _set_dataset_defaults(args)
    if getattr(args, "random_init_only", False):
        args.search_mode = "random"
        args.evolution_max_iter = 0
        logging.info("Random init only: no evolution; best structure = best of initial random population.")
    use_proxy = args.zero_shot_score is not None and args.zero_shot_score.strip() != ""
    search_mode = getattr(args, "search_mode", "llm")
    if getattr(args, "no_llm", False):
        search_mode = "mutation"
    use_llm = use_proxy and search_mode == "llm"
    logging.info("Search mode: %s (use_llm=%s)", search_mode, use_llm)

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
    np.random.seed(args.seed)
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

    # Initialize full population from scratch using random sampling (paper setting)
    logging.info("Initializing population of %d with random sampling from NAS-Bench-201 search space.", args.population_size)
    for i in range(args.population_size):
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
        if not np.isfinite(score):
            score = -9999.0
        popu_structure_list.append(arch_str)
        popu_index_list.append(arch_index if arch_index is not None else -1)
        popu_score_list.append(score)
        if (i + 1) % 50 == 0 or i == 0:
            logging.info("Init %d/%d done, current max score %.4f", i + 1, args.population_size, max(popu_score_list))
    min_safe = min(popu_score_list) if popu_score_list else 0
    if not np.isfinite(min_safe):
        min_safe = -9999.0
    if args.evolution_max_iter == 0:
        logging.info("Initial population ready. Max score=%.4f, min score=%.4f. No evolution (random_init_only or evolution_max_iter=0); returning best from population.", max(popu_score_list), min_safe)
    else:
        logging.info("Initial population ready. Max score=%.4f, min score=%.4f. Starting evolution.", max(popu_score_list), min_safe)

    for loop_count in range(args.evolution_max_iter):
        while len(popu_structure_list) > args.population_size:
            min_score = min(popu_score_list)
            idx_remove = popu_score_list.index(min_score)
            popu_score_list.pop(idx_remove)
            popu_structure_list.pop(idx_remove)
            popu_index_list.pop(idx_remove)

        if loop_count >= 1 and loop_count % 1000 == 0:
            scores_finite = [s for s in popu_score_list if np.isfinite(s)]
            max_s = max(scores_finite) if scores_finite else 0
            min_s = min(scores_finite) if scores_finite else -9999.0
            elapsed = time.time() - start_timer
            logging.info(
                "loop_count=%s/%s, max_score=%.4f, min_score=%.4f, time=%.2fh",
                loop_count, args.evolution_max_iter, max_s, min_s, elapsed / 3600,
            )

        # Propose next candidate: random search, LLM search, or mutation (evolution)
        if len(popu_structure_list) < 2:
            arch_str = get_random_arch_str(api)
        elif search_mode == "random":
            arch_str = get_random_arch_str(api)
        elif search_mode == "llm" and random.random() < getattr(args, "llm_prob", 0.5):
            # RZ-NAS: LLM proposes a mutated architecture (pick parent from top half by score)
            idx_sorted = sorted(range(len(popu_score_list)), key=lambda i: popu_score_list[i], reverse=True)
            top_half = max(1, len(idx_sorted) // 2)
            parent_idx = random.choice(idx_sorted[:top_half])
            parent_str = popu_structure_list[parent_idx]
            parent_score = popu_score_list[parent_idx]
            proxy_name = getattr(args, "zero_shot_score", "proxy") or "proxy"
            arch_str = get_new_structure_str_with_llm(parent_str, parent_score, proxy_name, api, args)
        else:
            # mutation mode, or llm_mode but not using LLM this step: mutate one op or random
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
        if not np.isfinite(score):
            score = -9999.0

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
