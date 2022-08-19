#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
  --model deit_small_patch16_224_384 \
  --epochs 300 \
  --batch-size 128 \
  --input-size 224 \
  --data-path /data/datasets/imagenet \
  --output_dir /data/runs/pss/scratch/224_384/baseline 
