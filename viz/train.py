import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()

SECS_TO_HOUR = 3600

ROOT = "../runs/"
runs = ["pretrain_lr_1.25e-3", "fracpatch_50p_rnd", "fracpatch_50p_mag", "fracpatch_random_linear", "baseline", "baseline-dense"]
for run in runs:
    train_path = os.path.join(ROOT, run, "train_log.json")
    train_data = load_jsonl(train_path)
    plt.plot(ema(train_data['loss'], 0.3), '-', linewidth=2, label=run)

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
    med_time = np.median(train_data["time"])
    timed_iters = [(i * med_time) / SECS_TO_HOUR for i in train_data["iteration"]]
    plt.plot(ema(timed_iters, 0.3), ema(train_data["loss"], 0.3), "-", linewidth=2, label=run)

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
    med_time = np.median(train_data["time"])
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

