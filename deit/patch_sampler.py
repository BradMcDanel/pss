import torch

def get_patch_idxs(x, drop_func, drop_ratio):
    if drop_func == "random":
        return get_random_patches(x, drop_ratio)
    elif drop_func == "magnitude":
        return get_magnitude_patches(x, drop_ratio)
    else:
        raise NotImplementedError

def get_random_patches(x, drop_ratio):
    B, N, C = x.shape
    keep_point = int(round((1-drop_ratio)*N)) - 1
    keep_patches = torch.rand(B, N-1, device=x.device).argsort(dim=1) + 1
    keep_patches = keep_patches[:, :keep_point]

    # do not drop cls patch
    cls_idx = torch.zeros((B, 1), device=keep_patches.device, dtype=keep_patches.dtype)

    batch_idxs = torch.arange(B, device=x.device).repeat_interleave(keep_point + 1)
    patch_idxs = torch.cat((cls_idx, keep_patches), dim=1).view(-1)

    return (batch_idxs, patch_idxs), (B, keep_point + 1)

def get_magnitude_patches(x, drop_ratio):
    B, N, C = x.shape
    keep_point = int(round((1-drop_ratio)*N)) - 1

    # sample patches using their magnitude
    sample_mat = x[:, 1:].abs().sum(dim=2)
    sample_mat = sample_mat / torch.sum(sample_mat, dim=1, keepdim=True)

    keep_patches = sample_mat.argsort(dim=1, descending=True) + 1
    keep_patches = keep_patches[:, :keep_point]

    # do not drop cls patch
    cls_idx = torch.zeros((B, 1), device=keep_patches.device, dtype=keep_patches.dtype)

    batch_idxs = torch.arange(B, device=x.device).repeat_interleave(keep_point + 1)
    patch_idxs = torch.cat((cls_idx, keep_patches), dim=1).view(-1)


    return (batch_idxs, patch_idxs), (B, keep_point + 1)
