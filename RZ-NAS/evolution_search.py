import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse, random, logging, time
import torch
from torch import nn
import numpy as np
import global_utils
import Masternet
import PlainNet
# from openai import AzureOpenAI
from dotenv import load_dotenv

# from tqdm import tqdm

from ZeroShotProxy import compute_zen_score, compute_te_nas_score, compute_syncflow_score, compute_gradnorm_score, compute_NASWOT_score
import benchmark_network_latency

working_dir = os.path.dirname(os.path.abspath(__file__))

def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--zero_shot_score', type=str, default=None,
                        help='could be: Zen (for Zen-NAS), TE (for TE-NAS)')
    parser.add_argument('--search_space', type=str, default=None,
                        help='.py file to specify the search space.')
    parser.add_argument('--evolution_max_iter', type=int, default=int(48e4),
                        help='max iterations of evolution.')
    parser.add_argument('--budget_model_size', type=float, default=None, help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
    parser.add_argument('--budget_flops', type=float, default=None, help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
    parser.add_argument('--budget_latency', type=float, default=None, help='latency of forward inference per mini-batch, e.g., 1e-3 means 1ms.')
    parser.add_argument('--max_layers', type=int, default=None, help='max number of layers of the network.')
    parser.add_argument('--batch_size', type=int, default=None, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--population_size', type=int, default=512, help='population size of evolution.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='output directory')
    parser.add_argument('--gamma', type=float, default=1e-2,
                        help='noise perturbation coefficient')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='number of classes')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

def generate_by_llm(structure_str):
    file_path = "prompt/template.txt"
    import os
    import openai

    load_dotenv(".env")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open(file_path, 'r') as file:
        prompt = file.read()
    # Use Chat Completions API (gpt-3.5-turbo / gpt-4 are chat models, not supported by v1/completions)
    kwargs = dict(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
        top_p=0.5,
    )
    if "<|im_end|>" in prompt:
        kwargs["stop"] = ["<|im_end|>"]
    response = openai.ChatCompletion.create(**kwargs)
    new_structure_str = response["choices"][0]["message"]["content"].strip()
    return new_structure_str

def _is_valid_structure(AnyPlainNet, structure_str, num_classes):
    """Return True if structure_str parses as a valid PlainNet structure."""
    try:
        s = (structure_str or '').strip()
        if not s:
            return False
        # Remove common LLM junk (markdown, extra text)
        for prefix in ('```', 'Here is', 'The structure is', 'Structure:'):
            if s.startswith(prefix) or s.lower().startswith(prefix.lower()):
                idx = s.find('SuperConv') if 'SuperConv' in s else s.find('Res')
                if idx >= 0:
                    s = s[idx:]
                    break
        if s.startswith('```'):
            s = s.split('```')[1].strip()
        net = AnyPlainNet(num_classes=num_classes, plainnet_struct=s, no_create=True)
        return hasattr(net, 'block_list') and getattr(net, 'block_list', None) is not None
    except Exception:
        return False

def get_new_random_structure_str(AnyPlainNet, structure_str, num_classes, get_search_space_func,
                                 num_replaces=1):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert isinstance(the_net, PlainNet.PlainNet)

    try:
        get_new_llm_structure_str = generate_by_llm(structure_str)
        # Strip to likely structure part (LLMs often add explanation)
        s = (get_new_llm_structure_str or '').strip()
        for prefix in ('```', 'Here is', 'The structure is', 'Structure:'):
            if s.lower().startswith(prefix.lower()):
                idx = s.find('SuperConv')
                if idx < 0:
                    idx = s.find('SuperRes')
                if idx < 0:
                    idx = s.find('ResBlock')
                if idx >= 0:
                    s = s[idx:]
                break
        if '```' in s:
            s = s.replace('```', '').strip()
        if _is_valid_structure(AnyPlainNet, s, num_classes):
            return s
    except Exception as e:
        logging.warning('LLM structure invalid or API error, using previous structure: %s', e)
    return structure_str


def get_splitted_structure_str(AnyPlainNet, structure_str, num_classes):
    try:
        the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
        if not hasattr(the_net, 'split') or not getattr(the_net, 'block_list', None):
            return structure_str
        splitted_net_str = the_net.split(split_layer_threshold=6)
        return splitted_net_str
    except Exception as e:
        logging.warning('Could not parse/split structure, using as-is: %s', e)
        return structure_str

def get_latency(AnyPlainNet, random_structure_str, gpu, args):
    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=False)
    if gpu is not None:
        the_model = the_model.cuda(gpu)
    the_latency = benchmark_network_latency.get_model_latency(model=the_model, batch_size=args.batch_size,
                                                              resolution=args.input_image_size,
                                                              in_channels=3, gpu=gpu, repeat_times=1,
                                                              fp16=True)
    del the_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return the_latency

def compute_nas_score(AnyPlainNet, random_structure_str, gpu, args):
    # compute network zero-shot proxy score
    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=True)
    if gpu is not None:
        the_model = the_model.cuda(gpu)
    try:
        if args.zero_shot_score == 'Zen':
            the_nas_core_info = compute_zen_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    mixup_gamma=args.gamma, batch_size=args.batch_size,
                                                                    repeat=1)
            the_nas_core = the_nas_core_info['avg_nas_score']
        elif args.zero_shot_score == 'TE-NAS':
            the_nas_core = compute_te_nas_score.compute_NTK_score(model=the_model, gpu=gpu,
                                                                  resolution=args.input_image_size,
                                                                  batch_size=args.batch_size)

        elif args.zero_shot_score == 'Syncflow':
            the_nas_core = compute_syncflow_score.do_compute_nas_score(model=the_model, gpu=gpu,
                                                                       resolution=args.input_image_size,
                                                                       batch_size=args.batch_size)

        elif args.zero_shot_score == 'GradNorm':
            the_nas_core = compute_gradnorm_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    batch_size=args.batch_size)

        elif args.zero_shot_score == 'Flops':
            the_nas_core = the_model.get_FLOPs(args.input_image_size)

        elif args.zero_shot_score == 'Params':
            the_nas_core = the_model.get_model_size()

        elif args.zero_shot_score == 'Random':
            the_nas_core = np.random.randn()

        elif args.zero_shot_score == 'NASWOT':
            the_nas_core = compute_NASWOT_score.compute_nas_score(gpu=gpu, model=the_model,
                                                                  resolution=args.input_image_size,
                                                                  batch_size=args.batch_size)
    except Exception as err:
        logging.info(str(err))
        logging.info('--- Failed structure: ')
        logging.info(str(the_model))
        # raise err
        the_nas_core = -9999


    del the_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return the_nas_core

def main(args, argv):
    gpu = args.gpu
    if gpu is not None and not torch.cuda.is_available():
        logging.warning('CUDA not available (CPU-only PyTorch). Running on CPU.')
        gpu = None
    if gpu is not None:
        torch.cuda.set_device('cuda:{}'.format(gpu))
        torch.backends.cudnn.benchmark = True

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    if os.path.isfile(best_structure_txt):
        print('skip ' + best_structure_txt)
        return None

    # load search space config .py file
    select_search_space = global_utils.load_py_module_from_path(args.search_space)

    # load masternet
    AnyPlainNet = Masternet.MasterNet

    masternet = AnyPlainNet(num_classes=args.num_classes, opt=args, argv=argv, no_create=True)
    initial_structure_str = str(masternet)

    popu_structure_list = []
    popu_zero_shot_score_list = []
    popu_latency_list = []

    start_timer = time.time()
    for loop_count in range(args.evolution_max_iter):
        # too many networks in the population pool, remove one with the smallest score
        while len(popu_structure_list) > args.population_size:
            min_zero_shot_score = min(popu_zero_shot_score_list)
            tmp_idx = popu_zero_shot_score_list.index(min_zero_shot_score)
            popu_zero_shot_score_list.pop(tmp_idx)
            popu_structure_list.pop(tmp_idx)
            popu_latency_list.pop(tmp_idx)
        pass

        if loop_count >= 1 and loop_count % 1000 == 0:
            max_score = max(popu_zero_shot_score_list)
            min_score = min(popu_zero_shot_score_list)
            elasp_time = time.time() - start_timer
            logging.info(f'loop_count={loop_count}/{args.evolution_max_iter}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time/3600:4g}h')

        # ----- generate a llm-to-nas structure ----- #
        if len(popu_structure_list) <= 10:
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=AnyPlainNet, structure_str=initial_structure_str, num_classes=args.num_classes,
                get_search_space_func=select_search_space.gen_search_space, num_replaces=1)
        else:
            tmp_idx = random.randint(0, len(popu_structure_list) - 1)
            tmp_random_structure_str = popu_structure_list[tmp_idx]
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=AnyPlainNet, structure_str=tmp_random_structure_str, num_classes=args.num_classes,
                get_search_space_func=select_search_space.gen_search_space, num_replaces=2)


        random_structure_str = get_splitted_structure_str(AnyPlainNet, random_structure_str,
                                                          num_classes=args.num_classes)

        the_model = None

# validate
        if args.max_layers is not None:
            if the_model is None:
                the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_layers = the_model.get_num_layers()
            if args.max_layers < the_layers:
                continue

        if args.budget_model_size is not None:
            if the_model is None:
                the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_size = the_model.get_model_size()
            if args.budget_model_size < the_model_size:
                continue

        if args.budget_flops is not None:
            if the_model is None:
                the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_flops = the_model.get_FLOPs(args.input_image_size)
            if args.budget_flops < the_model_flops:
                continue

        the_latency = np.inf
        if args.budget_latency is not None:
            the_latency = get_latency(AnyPlainNet, random_structure_str, gpu, args)
            if args.budget_latency < the_latency:
                continue
        the_nas_core = compute_nas_score(AnyPlainNet, random_structure_str, gpu, args)
        # import pdb
        # pdb.set_trace()
        popu_structure_list.append(random_structure_str)
        popu_zero_shot_score_list.append(the_nas_core)
        popu_latency_list.append(the_latency)

    return popu_structure_list, popu_zero_shot_score_list, popu_latency_list






if __name__ == '__main__':
    args = parse_cmd_options(sys.argv)
    log_fn = os.path.join(args.save_dir, 'evolution_search.log')
    global_utils.create_logging(log_fn)

    info = main(args, sys.argv)
    if info is None:
        exit()



    popu_structure_list, popu_zero_shot_score_list, popu_latency_list = info

    # export best structure
    best_score = max(popu_zero_shot_score_list)
    best_idx = popu_zero_shot_score_list.index(best_score)
    best_structure_str = popu_structure_list[best_idx]
    the_latency = popu_latency_list[best_idx]

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    global_utils.mkfilepath(best_structure_txt)
    with open(best_structure_txt, 'w') as fid:
        fid.write(best_structure_str)
    pass  # end with

    # export top 50 structures with scores and latencies
    top_k = 50
    indexed = list(zip(popu_zero_shot_score_list, popu_structure_list, popu_latency_list))
    indexed.sort(key=lambda x: x[0], reverse=True)
    top50 = indexed[:top_k]
    top50_txt = os.path.join(args.save_dir, 'top50_structures.txt')
    global_utils.mkfilepath(top50_txt)
    with open(top50_txt, 'w') as fid:
        fid.write('# Top {} structures: rank | zero_shot_score | latency\n'.format(top_k))
        fid.write('# rank\tzero_shot_score\tlatency\tstructure\n')
        for rank, (score, structure, latency) in enumerate(top50, start=1):
            fid.write('{}\t{}\t{}\t{}\n'.format(rank, score, latency, structure))
    pass  # end with
