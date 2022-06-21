import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()

SECS_TO_HOUR = 3600

ROOT = "/data/runs/simmim/fracpatch_finetune/"
runs = ["baseline", "random_fixed_50", "random_linear_80_0"]
for run in runs:
    train_path = os.path.join(ROOT, run, "train_log.json")
    train_data = load_jsonl(train_path)
    plt.plot(ema(train_data['loss'], 0.9999), '-', linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Training Iteration')
plt.ylabel('Training Loss')
plt.legend(loc=0)
plt.savefig('../figures/training-loss.pdf', dpi=300, bbox_inches='tight')
plt.clf()

for run in runs:
    val_path = os.path.join(ROOT, run, "val_log.json")
    val_data = load_jsonl(val_path)
    plt.plot(val_data['acc1'], '-', linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend(loc=0)
plt.savefig('../figures/validation-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()

# accounting for running time
for run in runs:
    train_path = os.path.join(ROOT, run, "train_log.json")
    train_data = load_jsonl(train_path)
    num_iters = float(train_data["iteration"][-1])
    med_time = np.mean(train_data["time"])
    timed_iters = [(i * med_time) / SECS_TO_HOUR for i in train_data["iteration"]]
    plt.plot(ema(timed_iters, 0.9999), ema(train_data["loss"], 0.9999), "-", linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Training Time (Hours)')
plt.ylabel('Training Loss')
plt.legend(loc=0)
plt.savefig('../figures/training-loss-time.pdf', dpi=300, bbox_inches='tight')
plt.clf()

for run in runs:
    train_path = os.path.join(ROOT, run, "train_log.json")
    train_data = load_jsonl(train_path)
    val_path = os.path.join(ROOT, run, "val_log.json")
    val_data = load_jsonl(val_path)
    num_iters = float(train_data["iteration"][-1])
    med_time = np.mean(train_data["time"])
    timed_iters = [(i * med_time) / SECS_TO_HOUR for i in train_data["iteration"]]
    idxs = np.rint(np.linspace(0, len(timed_iters)-1, len(val_data["acc1"]))).astype('int')
    epoch_times = [timed_iters[i] for i in idxs]
    plt.plot(epoch_times, val_data["acc1"], "-", linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Training Time (Hours)')
plt.ylim((65, 84))
plt.ylabel('Validation Accuracy')
plt.legend(loc=0)
plt.savefig('../figures/validation-accuracy-time.pdf', dpi=300, bbox_inches='tight')
plt.clf()


for run in runs:
    val_path = os.path.join(ROOT, run, "val_log.json")
    val_data = load_jsonl(val_path)
    print(f'Max val accuray for {run}: {max(val_data["acc1"])}')
