import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()

SECS_TO_HOUR = 3600

ROOT = "/data/runs/"
runs = ["dynamicvit/deit-s", "fracpatch/scratch/224_384/magnitude_cyclic_80_0"]
names = ["DynamicViT-S", "DieT-S+PSS"]
colors = ["#6A3D9A", "#E31A1C"]
hashes = ['o','s']
plt.figure(figsize=(8, 4))
for i, run in enumerate(runs):
    train_path = os.path.join(ROOT, run, "sweep_drop.json")
    data = load_jsonl(train_path)
    keep_ratios = [1 - drop_ratio for drop_ratio in data["drop_ratio"]]
    plt.plot(keep_ratios, data["acc1"],
             label=names[i], color=colors[i], marker=hashes[i], linewidth=2)

plt.title('Accuracy vs Throughput')
plt.xlabel('Throughput (images/sec)')
plt.ylabel('Validation Acc. (%)')
# plt.ylim(0.6, 1.0)
plt.ylim((70, 80))
# plt.xlim((0.375, 1.025))
# plt.gca().invert_xaxis()
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('../figures/dynamicvit-comp.pdf', dpi=300, bbox_inches='tight')
plt.clf()
