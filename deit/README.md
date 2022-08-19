# DieT + PSS

This is a fork of the [DieT repository](https://github.com/facebookresearch/deit). Many of the files are not used, as we only explored PSS applied to the initial DieT version. The main modifications to the project are:
- `pss.py`: This is the main implementation of DeiT models with the inclusion of a Patch Sampling Scheduele. The patch sorting functions can be found near the top of the file. As noted in the paper, we include relative position embedding, as we find it dramatically improves the performance of ViT models.
- `models.py`: We added several models based on the baseline deit models. For instance, `pss_deit_small_patch16_224_384` generates a DeiT-S model with PSS included. We modified the names of the deit models in the original codebase to distinguish the two resolutions of DeiT-S models we trained in our work (224x224 and 384x384). 
- `patch_scheduler.py`: This script implements the patch sampling schedules mentioned in the paper.
- `main.py`: We modified this main training script to support the use of a PSS. The PSS is updated per training iteration using a `.step()` function.

## Training
First, check the `configs/` directory for training settings that are evaluated in the paper. For instance, the `configs/224_384` directory contains `baseline.sh` (DeiT-S-224) and `magnitude_cyclic_80_0.sh` (DeiT-S-224+PSS) scripts. The `data-path` and `output_dir` in each script should be modified according to your system.

From the `simmim/` directory, do the following to train each model:

### DieT-S-224
`bash configs/224_384/baseline.sh`

### DieT-S-224 + PSS
`bash configs/224_384/magnitude_cyclic_80_0.sh`

### DieT-S-384
`bash configs/384_384/baseline.sh`

### DieT-S-384 + PSS
`bash configs/384_384/magnitude_cyclic_80_0.sh`

This will generate an output directory specified by `output_dir` which will store a `train_log.json` (updated per training iteration) and a `val_log.json` (updated per epoch). Additionally, it will save a `checkpoint.pth` for the last epoch and a `best_checkpoint.pth` for the best checkpoint found so far. In our evalutations, we use the `best_checkpoint.pth` for each model.

## Validate
You should be able to use the `--eval` with `main.py` as in the DeiT repository.

## Experiments
Here, we show how to generate data used for some of the figures in the paper. Refer to the `viz/` directory (in root of codebase) to see how these generated data files are used.

- `image_patches.py`: This is used to generate image patch data (which patches are selected) for a given model and `\rho` combination. This will generate a `image_patches.json` file in the same directory as the checkpoint (`--resume`). To run this file, do:
```
python image_patches.py --model pss_deit_small_patch16_224_384 \
  --resume <checkpoint.pth path> \
  --data-path <root imagenet path> \
  --input-size 224 \
  --batch-size 256 \
  --num_workers 32
```
- `sweep_drop.py`: This generated the dynamic inference curve that varies `\rho` to get an accuracy trade-off. Note that `num_workers` is set to 32 in order to ensure the model is not CPU bottlenecked. This will generate a `image_patches.json` file in the same directory as the checkpoint (`--resume`). To run this file, do:
```
python sweep_drop.py --model pss_deit_small_patch16_224_384 \
  --resume <checkpoint.pth path>
  --data-path <root imagenet path> \
  --input-size 224 \
  --batch-size 256 \
  --num_workers 32
```
