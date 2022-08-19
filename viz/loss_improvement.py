import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()


SECS_TO_HOUR = 3600

ROOT = "/data/runs/pss/finetune/vit-b"
runs = ["baseline", "magnitude_cyclic_80_0"]
names = ["ViT-B", "+PSS"]
colors = ["#1F78B4", "#E31A1C"]
hashes = ['o', 's']
train_datas, val_datas, sweep_datas = {}, {}, {}
for run in runs:
    train_datas[run] = load_jsonl(os.path.join(ROOT, run, "train_log.json"))
    val_datas[run] = load_jsonl(os.path.join(ROOT, run, "val_log.json"))


# for cyclic, compute the loss improvement per epoch and drop_ratio
# for baseline, compute the loss improvement per epoch
drop_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
train_drop_ratios = np.array(train_datas["magnitude_cyclic_80_0"]["patch_drop_ratio"])
train_losses = np.array(train_datas["magnitude_cyclic_80_0"]["loss"])

i = 20
train_drop_ratios = train_drop_ratios[i*2502:(i+5)*2502]
train_losses =  train_losses[i*2502:(i+5)*2502]

train_loss_diffs = train_losses[:-1] - train_losses[1:]
# prepend 0.0 to the loss_diffs
train_loss_diffs = np.concatenate([[train_loss_diffs[0]], train_loss_diffs])
drop_ratio_losses = []
# # find outliers (caused by shifts in patch drop ratio)
idxs = np.abs(train_loss_diffs - np.mean(train_loss_diffs)) < 1 * np.std(train_loss_diffs)
train_loss_diffs = train_loss_diffs[idxs]
train_drop_ratios = train_drop_ratios[idxs]


# normalize loss diff by drop_ratio
# train_loss_diffs = train_loss_diffs / (1 - train_drop_ratios)

plt.plot(train_loss_diffs, '-', linewidth=2, label="Train Loss Improvement", color=colors[1])
plt.savefig("../figures/loss-improvement.pdf")
plt.clf()

# for drop_ratio in drop_ratios:
#     # compute indexs of drop_ratio in train_drop_ratios
#     idxs = np.where(train_drop_ratios == drop_ratio)[0]

#     drop_ratio_losses.append(np.mean(train_loss_diffs[idxs]))

# for drop_ratio, loss_diff in zip(drop_ratios, drop_ratio_losses):
#     print("drop_ratio={:.1f}, loss_diff={:.5f}".format(drop_ratio, loss_diff))



