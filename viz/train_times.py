import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()

SECS_TO_HOUR = 3600

ROOT = "/data/runs/fracpatch/finetune/vit-b"
runs = ["magnitude_cyclic_80_0"]
for run in runs:
    train_path = os.path.join(ROOT, run, "train_log.json")
    data = load_jsonl(train_path)
    times = data["time"][1:2502]
    print(run, sum(times))
    # times = np.clip(times, 0, np.percentile(times, 99))
    # times = ema(times, 0.99)

    plt.plot(times, '-', linewidth=2, label=run)


plt.title('ImageNet (VIT)')
plt.xlabel('Iteration')
plt.ylabel('Iteration Time (s)')
plt.legend(loc=0)
plt.savefig('../figures/train-times.pdf', dpi=300, bbox_inches='tight')
plt.clf()
