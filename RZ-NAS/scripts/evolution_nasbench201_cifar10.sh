#!/bin/bash
# Paper setting: evolution on NAS-Bench-201 with proxy (e.g. Zen), then train best arch 3 runs.
# No benchmark .pth file required for search.

cd "$(dirname "$0")/.."

PROXY="${PROXY:-Zen}"
SAVE_DIR="${SAVE_DIR:-./save_dir/nasbench201_${PROXY}}"
mkdir -p "$SAVE_DIR"

# 1) Search with zero-shot proxy
python evolution_search_nasbench201.py \
  --zero_shot_score "$PROXY" \
  --evolution_max_iter 50000 \
  --population_size 256 \
  --batch_size 64 --input_image_size 32 --num_classes 10 \
  --gpu 0 \
  --save_dir "$SAVE_DIR"

# 2) Train best architecture 3 runs (benchmark setting)
python train_nasbench201_3runs.py \
  --plainnet_struct_txt "$SAVE_DIR/best_structure.txt" \
  --save_dir "$SAVE_DIR/train_3runs" \
  --dataset cifar10 --num_classes 10 \
  --epochs 200 --batch_size 128 --gpu 0
