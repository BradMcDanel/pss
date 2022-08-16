#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
  --model fracpatch_deit_small_patch16_224_384 \
  --epochs 300 \
  --batch-size 256 \
  --input-size 224 \
  --data-path /data/datasets/imagenet \
  --output_dir /data/runs/fracpatch/scratch/224_384/magnitude_cyclic_80_0 \
  --patch-scheduler-name cyclic \
  --patch-scheduler-start-drop-ratio 0.8 \
  --patch-scheduler-end-drop-ratio -0.1 \
  --patch-drop-func magnitude
