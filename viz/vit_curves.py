import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()


SECS_TO_HOUR = 3600


ROOT = "/data/runs/fracpatch/finetune/vit-b"
runs = ["baseline", "magnitude_fixed_40",
        "magnitude_linear_80_0", "magnitude_cyclic_80_0"]
names = ["ViT-B", "+Fixed(0.6)", "+Linear(0.2, 1.0)", "+Cyclic(0.2, 1.0)"]
colors = ["#1F78B4", "#FF7F0F", "#33A02C", "#E31A1C"]

train_datas, val_datas = {}, {}
for run in runs:
    train_path = os.path.join(ROOT, run, "train_log.json")
    val_path = os.path.join(ROOT, run, "val_log.json")
    train_datas[run] = load_jsonl(train_path)
    val_datas[run] = load_jsonl(val_path)

# create plot with two column subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for i, run in enumerate(runs):
    loss = train_datas[run]["loss"]
    ax[0].plot(ema(loss, 0.9999), '-', linewidth=2, label=names[i], color=colors[i])

ax[0].set_xlabel('Training Iteration')
# set tick labels to scientific notation
ax[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax[0].set_ylabel('Training Loss')
# plt.legend(loc=0)

  
for i, run in enumerate(runs):
    total_time = np.cumsum(train_datas[run]["time"]) / SECS_TO_HOUR
    ax[1].plot(ema(total_time, 0.9999), ema(train_datas[run]["loss"], 0.9999), "-", linewidth=2, label=names[i], color=colors[i])


ax[1].set_xlabel('Training Time (Hours)')
# plt.ylabel('Training Loss')
# legend
leg = ax[1].legend(loc=0)
plt.savefig('../figures/vit-training-loss.pdf', dpi=300, bbox_inches='tight')
plt.clf()


fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for i, run in enumerate(runs):
    ax[0].plot(val_datas[run]["loss"], "-", linewidth=2, label=names[i], color=colors[i])

ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Validation Loss')

for i, run in enumerate(runs):
    total_time = np.cumsum(train_datas[run]["time"]) / SECS_TO_HOUR
    idxs = np.rint(np.linspace(0, len(total_time)-1, len(val_datas[run]["loss"]))).astype('int')
    epoch_times = [total_time[j] for j in idxs]
    ax[1].plot(epoch_times, val_datas[run]["loss"], "-", linewidth=2, label=names[i], color=colors[i])

ax[1].set_xlabel('Training Time (Hours)')
plt.legend(loc=0)
plt.savefig('../figures/vit-validation-loss.pdf', dpi=300, bbox_inches='tight')
plt.clf()


fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for i, run in enumerate(runs):
    ax[0].plot(val_datas[run]["acc1"], "-", linewidth=2, label=names[i], color=colors[i])

ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Validation Accuracy (%)')

for i, run in enumerate(runs):
    total_time = np.cumsum(train_datas[run]["time"]) / SECS_TO_HOUR
    idxs = np.rint(np.linspace(0, len(total_time)-1, len(val_datas[run]["loss"]))).astype('int')
    epoch_times = [total_time[j] for j in idxs]
    ax[1].plot(epoch_times, val_datas[run]["acc1"], "-", linewidth=2, label=names[i], color=colors[i])

ax[1].set_xlabel('Training Time (Hours)')
plt.legend(loc=0)
plt.savefig('../figures/vit-validation-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()
