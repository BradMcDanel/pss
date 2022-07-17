import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()

SECS_TO_HOUR = 3600

ROOT = "/data/runs/deit"
runs = ["small-baseline-v2", "small-cyclic-80-0-magnitude-v3", "small-multilayer-cyclic"]
for run in runs:
    train_path = os.path.join(ROOT, run, "train_log.json")
    data = load_jsonl(train_path)
    times = data["gpu_time"][:5000]
    # trim bottom 1% and top 1%
    times = np.clip(times, 0, np.percentile(times, 99))
    times = ema(times, 0.9)

    plt.plot(times, '-', linewidth=2, label=run)


plt.title('ImageNet (VIT)')
plt.xlabel('Iteration')
plt.ylabel('Iteration Time (s)')
plt.legend(loc=0)
plt.savefig('../figures/train-times.pdf', dpi=300, bbox_inches='tight')
plt.clf()
