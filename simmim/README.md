# SimMIM + PSS

This is a fork of the [SimMIM repository](https://github.com/microsoft/SimMIM). Many of the files are not used, as we only explored PSS applied to the initial ViT-B version (not Swin Transformer). The main modifications to the project are:
- `models/pss_vision_transformer.py`: This is the main implementation of ViT-B models with the inclusion of a Patch Sampling Scheduele. The patch sorting functions can be found near the top of the file. As noted in the paper, we include relative position embedding, as we find it dramatically improves the performance of ViT models.
- Configuration changes: We had to make minor changes in many files to make the `PSS` model visible to the training script.
- `patch_scheduler.py`: This script implements the patch sampling schedules mentioned in the paper.
- `main_finetune.py`: We modified this main training script to support the use of a PSS. The PSS is updated per training iteration using a `.step()` function.

## Training
First, check the `configs/vit-b/` directory for training settings that are evaluated in the paper. For instance, the `configs/vit-b/baseline.yaml` trains a baseline (ViT-B-224) model and `configs/vit-b/magnitude_cyclic_80_0.yaml` trains a (ViT-B-224+PSS) model.

From the `simmim/` directory, do the following to train each model:

### ViT-B-224
```
python -m torch.distributed.launch --nproc_per_node 4 main_finetune.py  \
  --cfg configs/vit-b/baseline.yaml \
  --data-path <root imagenet path> \
  --pretrained <pretrained ViT-B SimMIM model> \
  --output <root output directory>
```

### ViT-B-224 + PSS - Cyclic(0.2, 1.0)
```
python -m torch.distributed.launch --nproc_per_node 4 main_finetune.py  \
  --cfg configs/vit-b/magnitude_cyclic_80_0.yaml \
  --data-path <root imagenet path> \
  --pretrained <pretrained ViT-B SimMIM model> \
  --output <root output directory>
```

### ViT-B-224 + PSS - Linear(0.2, 1.0)
```
python -m torch.distributed.launch --nproc_per_node 4 main_finetune.py  \
  --cfg configs/vit-b/magnitude_linear_80_0.yaml \
  --data-path <root imagenet path> \
  --pretrained <pretrained ViT-B SimMIM model> \
  --output <root output directory>
```

All other model in the `config/` directory work in a similar fashion.

This will generate an output directory specified by `output` which will store a `train_log.json` (updated per training iteration) and a `val_log.json` (updated per epoch). Additionally, it will save a `checkpoint.pth` every 25 epochs.

## Validate
You should be able to use the `--eval` with `main.py` as in the DeiT repository.

## Experiments
Here, we show how to generate data used for some of the figures in the paper. Refer to the `viz/` directory (in root of codebase) to see how these generated data files are used.

- `image_patches.py`: This is used to generate image patch data (which patches are selected) for a given model and `\rho` combination. This will generate a `image_patches.json` file in the same directory as the checkpoint (`--resume`). To run this file, do:
```
python -m torch.distributed.launch --nproc_per_node 1 image_patches.py \
  --cfg <config path used in training> \
  --data-path <root imagenet path> \
  --pretrained <checkpoint.pth path> \
  --output <root output directory>
```
- `sweep_drop.py`: This generated the dynamic inference curve that varies `\rho` to get an accuracy trade-off. Note that `num_workers` is set to 32 in order to ensure the model is not CPU bottlenecked. This will generate a `image_patches.json` file in the same directory as the checkpoint (`--resume`). To run this file, do:
```
python -m torch.distributed.launch --nproc_per_node 1 sweep_drop.py \
  --cfg <config path used in training> \
  --data-path <root imagenet path> \
  --pretrained <checkpoint.pth path> \
  --output <root output directory>
```
