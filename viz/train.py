import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()


SECS_TO_HOUR = 3600


ROOT = "/data/runs/fracpatch/finetune/vit-b/"
# runs = ["baseline", "magnitude_cyclic_80_0", "random_cyclic_80_0", "magnitude_fixed_40"]
runs = ["old", "magnitude_cyclic_80_0"]
# ROOT = "/data/runs/deit"
# runs = ["small-baseline-v2", "small-cyclic-80-0-magnitude-ft-v2"]
train_datas, val_datas = {}, {}
for run in runs:
    train_path = os.path.join(ROOT, run, "train_log.json")
    val_path = os.path.join(ROOT, run, "val_log.json")
    train_datas[run] = load_jsonl(train_path)
    val_datas[run] = load_jsonl(val_path)

for run in runs:
    plt.plot(ema(train_datas[run]['loss'], 0.9999), '-', linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Training Iteration')
plt.ylabel('Training Loss')
plt.legend(loc=0)
plt.savefig('../figures/training-loss.pdf', dpi=300, bbox_inches='tight')
plt.clf()

for run in runs:
    plt.plot(val_datas[run]['acc1'], '-', linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend(loc=0)
plt.savefig('../figures/validation-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()

# accounting for running time
for run in runs:
    total_time = np.cumsum(train_datas[run]["gpu_time"]) / SECS_TO_HOUR
    plt.plot(ema(total_time, 0.9999), ema(train_datas[run]["loss"], 0.9999), "-", linewidth=2, label=run)
    print(np.sum(train_datas[run]["gpu_time"]) / SECS_TO_HOUR, run)

plt.title('ImageNet (VIT)')
plt.xlabel('Training Time (Hours)')
plt.ylabel('Training Loss')
plt.legend(loc=0)
plt.savefig('../figures/training-loss-time.pdf', dpi=300, bbox_inches='tight')
plt.clf()

for run in runs:
    total_time = np.cumsum(train_datas[run]["gpu_time"]) / SECS_TO_HOUR
    idxs = np.rint(np.linspace(0, len(total_time)-1, len(val_datas[run]["acc1"]))).astype('int')
    epoch_times = [total_time[i] for i in idxs]
    plt.plot(epoch_times, val_datas[run]["acc1"], "-", linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Training Time (Hours)')
plt.ylim((65, 84))
plt.ylabel('Validation Accuracy')
plt.legend(loc=0)
plt.savefig('../figures/validation-accuracy-time.pdf', dpi=300, bbox_inches='tight')
plt.clf()


for run in runs:
    plt.plot(train_datas[run]["lr"], "-", linewidth=2, label=run)

plt.title("ImageNet (VIT)")
plt.xlabel('Iteration')
# plt.ylim((65, 84))
plt.ylabel('Learning Rate')
plt.legend(loc=0)
plt.savefig('../figures/learning-rates.pdf', dpi=300, bbox_inches='tight')
plt.clf()


for run in runs:
    print(f'Max val accuray for {run}: {max(val_datas[run]["acc1"])}')
