import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()

SECS_TO_HOUR = 3600

ROOT = "/data/runs/deit"
runs = ["small-baseline-v2", "small-cyclic-80-0-magnitude-v3", "small-cyclic-80-0-magnitude-ft-v2"]
for run in runs:
    train_path = os.path.join(ROOT, run, "sweep_drop.json")
    data = load_jsonl(train_path)
    plt.plot([100. * d for d in data["drop_ratio"]], data["acc1"], '-o', linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Patch Drop (%)')
plt.ylabel('Validation Accuracy')
plt.legend(loc=0)
plt.savefig('../figures/sweep-drop-ratio.pdf', dpi=300, bbox_inches='tight')
plt.clf()
