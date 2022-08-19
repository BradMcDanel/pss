#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
  --model deit_small_patch16_224_384 \
  --epochs 300 \
  --batch-size 256 \
  --input-size 224 \
  --data-path "$1" \
  --output_dir "$2"/pss/deit/deit-s-224/baseline 
