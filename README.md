# Accelerating Vision Transformer Training via a Patch Sampling Schedule (PSS)

This codebase shows how to train and validate models trained with a Patch Sampling Scheduele (PSS).

## Model Checkpoints and Logs
| **Codebase** | **Model**        | **Checkpoint**                                                                                   | **Logs**                                                                                   | **Pretrained?** | **Training Time** | **Accuracy** |
|--------------|------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|-----------------|-------------------|--------------|
| SimMIM       | ViT-B            | [checkpoint](https://drive.google.com/file/d/1Y90a-1TDlTH7v3yqGMaTdSnRuFKiwka3/view?usp=sharing) | [logs](https://drive.google.com/file/d/1NY1Aw2E8MSOKuUiNmNRD6vJ8eavdL-YS/view?usp=sharing) | yes             | 24.6h             | 83.7%        |
| SimMIM       | ViT-B + PSS      | [checkpoint](https://drive.google.com/file/d/1rrWrKTjZdd2nYSR-AuS6cktEiBFGgGvq/view?usp=sharing) | [logs](https://drive.google.com/file/d/1OzXM8CWqWTkfel8ht_K71PHsXx3grZUI/view?usp=sharing) | yes             | 17.0h             | 83.5%        |
| DeiT         | DeiT-S-224       | [checkpoint](https://drive.google.com/file/d/1AVYJlA97mfQEZkUl0vjdEQGWdfO2HpmK/view?usp=sharing) | [logs](https://drive.google.com/file/d/1u6-Jb8R4G7NU62za8sRl-muKimpsYtU-/view?usp=sharing) | no              | 29.9h             | 80.1%        |
| DeiT         | DeiT-S-224 + PSS | [checkpoint](https://drive.google.com/file/d/1Afp8S26hBWsyhmX6aztrcFhOaL5FoT6-/view?usp=sharing) | [logs](https://drive.google.com/file/d/1eQlj1zTRdtQd2yryH-9LT247M3uXHtFJ/view?usp=sharing) | no              | 27.9h             | 80.4%        |
| DeiT         | Deit-S-384       | Training                                                                                         | Training                                                                                   | no              | ~150h             | ?            |
| DeiT         | DeiT-S-384 + PSS | [checkpoint](https://drive.google.com/file/d/1QxKDws4b9GdZwhp-FnrKg967Ku5ZJigG/view?usp=sharing) | [logs](https://drive.google.com/file/d/1u0hCJWPAjSsDtnLdBvo9foZcWCJ58Fw4/view?usp=sharing) | no              | 109.0h            | 82.7%        |


## Training and Inference results
Please refer to the `simmim/` or `deit/` directories for more information on how to train and validate PSS models.

The `data/` directory provides a snapshot of the results for both ViT-B and DeiT models.
