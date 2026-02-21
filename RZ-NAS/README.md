# RZ-NAS: Enhancing LLM-guided Neural Architecture Search via Reflective Zero-Cost Strategy

## About
This is the repository for the paper RZ-NAS: Enhancing LLM-guided Neural Architecture Search via Reflective Zero-Cost Strategy. 

LLM-to-NAS is a promising field at the intersection of Large Language Models (LLMs) and Neural Architecture Search (NAS), as recent research has explored the potential of architecture generation leveraging LLMs on multiple search spaces. However, the existing LLM-to-NAS methods face the challenges of limited search spaces, time-cost search efficiency, and uncompetitive performance across standard NAS benchmarks and multiple downstream tasks. In this work, we propose Reflective Zero-cost NAS (RZ-NAS) that can search NAS architectures with humanoid reflections and training-free metrics to elicit the power of LLMs. We rethink LLMs' roles in NAS in current work and design a structured, prompt-based to comprehensively understand the search task and architectures from both text and code levels. By integrating LLM reflection modules, we use LLM-generated feedback to provide linguistic guidance within architecture optimization. RZ-NAS enables effective search within both micro and macro search spaces without extensive time cost, achieving SOTA performance across multiple downstream tasks. 


## Search

One example of the whole prompt is saved in `template.txt`. More zero-cost proxies and search spaces are saved under the folder `descriptions`. 


For different zero-cost proxies, you can change the parameter `zero_shot_score`.

```
python evolution_search.py --gpu 0 --zero_shot_score <zero-cost proxy> --search_space <micro/macro search space> 
```

more customized parameters setting can be found in ./scripts.

### NAS-Bench-201 search space (paper setting: proxy + train 3 runs)

To compare accuracy on NAS-Bench-201 as in the paper: **search with a zero-shot proxy**, then **train the best-scoring network for three runs** under the benchmark training setting and report mean ± std test accuracy.

1. **Search** (fitness = proxy score; no benchmark file needed):
   ```bash
   python evolution_search_nasbench201.py \
     --zero_shot_score Zen \
     --evolution_max_iter 50000 \
     --population_size 256 \
     --batch_size 64 --input_image_size 32 --num_classes 10 \
     --gpu 0 \
     --save_dir ./save_dir/nasbench201_zen
   ```
   Use `--zero_shot_score` one of: `Zen`, `TE-NAS`, `Syncflow`, `GradNorm`, `NASWOT`, `Flops`, `Params`, `Random`. Output: `best_structure.txt` (NAS-Bench-201 arch string).

2. **Train the best architecture 3 runs** (benchmark setting: 200 epochs, SGD, CIFAR augmentation):
   ```bash
   python train_nasbench201_3runs.py \
     --plainnet_struct_txt ./save_dir/nasbench201_zen/best_structure.txt \
     --save_dir ./save_dir/nasbench201_zen/train_3runs \
     --dataset cifar10 --num_classes 10 \
     --epochs 200 --batch_size 128 --gpu 0
   ```
   Result: `train_3runs_result.txt` with test accuracy mean ± std over 3 runs.

**Optional – precomputed fitness:** If you omit `--zero_shot_score` and pass `--benchmark_path` to a downloaded `NAS-Bench-201-v1_1-096897.pth` ([NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)), evolution uses the benchmark’s validation/test accuracy as fitness instead of a proxy.



