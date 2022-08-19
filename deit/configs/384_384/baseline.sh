#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
  --model deit_small_patch16_384_384 \
  --epochs 300 \
  --batch-size 128 \
  --input-size 384 \
  --data-path /data/datasets/imagenet \
  --output_dir /data/runs/pss/scratch/384_384/baseline 
