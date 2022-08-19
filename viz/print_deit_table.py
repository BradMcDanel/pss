import os
import numpy as np

from utils import load_jsonl, ema

SECS_TO_HOUR = 3600

ROOT = "/data/runs/pss/scratch/"
runs = ["224_384/baseline", "224_384/magnitude_cyclic_80_0", "384_384/magnitude_cyclic_80_0"]

train_datas, val_datas = {}, {}
for run in runs:
    train_path = os.path.join(ROOT, run, "train_log.json")
    val_path = os.path.join(ROOT, run, "val_log.json")
    train_datas[run] = load_jsonl(train_path)
    val_datas[run] = load_jsonl(val_path)


for i, (train_data, val_data) in enumerate(zip(train_datas.values(), val_datas.values())):
    train_time = np.sum(train_data["time"]) / SECS_TO_HOUR
    gpu_time = np.sum(train_data["gpu_time"]) / SECS_TO_HOUR
    val_acc = val_data["acc1"][-1]

    print(f"{runs[i]}. {train_time:.2f}h total, {gpu_time:.2f}h gpu, {val_acc:.2f}")
