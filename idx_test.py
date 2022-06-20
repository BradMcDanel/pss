import torch

x = torch.arange(0, 160).view(2, 5, 4, 4)
idxs_0 = torch.tensor([0, 0, 1, 1])
idxs_2 = torch.tensor([0, 1, 0, 1])
sel = x[idxs_0, :, idxs_2]
sel = sel.view(2, 2, 5, 4).permute(0, 2, 1, 3).contiguous()
sel = sel[idxs_0, :, :, idxs_2]
sel = sel.view(2, 2, 5, 2).permute(0, 2, 3, 1).contiguous()
print(x)
print(sel)
print(sel.shape)
