# Accelerating Vision Transformer Training via a Patch Sampling Schedule (PSS)

This codebase shows how to train and validate models trained under PSS. During development of this work, we named the model FracPatch internally. While this name does not appear in our PSS paper, a FracPatch model is simply the underlying ViT model with the addition of the Patch Sampling Block after the patch embedded layer and the ability to toggle the patch keep rate `\rho` during each iteration.

**AAAI Notes**:
- Each DeiT-S model is around 350MB, which means we were unable to upload a single model as part of the submission.
- As mentioned in the paper, we apply PSS to both SimMIM (`simmim/`) and DeiT (`deit`) models. Thus, we include both repositories as subdirectories in our codebase. Currently, we use duplicated files for PSS related code, but this will eventually be refactored into a standalone module.
- During training, we collect statistics in a json file call `train_log.json`. However, this file is around 150MB for a single training (for DeiT-S). Therefore, for have subsampled these training files so that you can review a snapshop of the training logs.
- We provide most of the data necessary to reproduce the figures generated in the main paper and the appendix. Refer to the `viz/` directory for information on how to run each script. These scripts generate `.pdf` files that are saved in the `figures/` directory.


## Training and Inference results
Please refer to the `simmim` or `deit` directories for more information on how to train and validate these models.

The `data/` directory provides a snapshot of the results for both ViT-B and DeiT models.
