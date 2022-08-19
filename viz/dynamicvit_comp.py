import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()

SECS_TO_HOUR = 3600

ROOT = "/data/runs/"
runs = ["dynamicvit/deit-s", "pss/scratch/224_384/magnitude_cyclic_80_0"]
names = ["DynamicViT-S", "DieT-S+PSS"]
colors = ["#6A3D9A", "#E31A1C"]
hashes = ['o','s']
plt.figure(figsize=(8, 5.5))
for i, run in enumerate(runs):
    train_path = os.path.join(ROOT, run, "sweep_drop.json")
    data = load_jsonl(train_path)
    keep_ratios = [1 - drop_ratio for drop_ratio in data["drop_ratio"]]
    plt.plot(keep_ratios, data["acc1"],
             label=names[i], color=colors[i], marker=hashes[i], linewidth=2)

    if i == 0:
        x = 0.725
        y = 77.5
        text = r"target $\rho$=0.7"
        plt.text(x, y, text, ha='center', va='bottom', fontsize=14)
        text = r"for DynamicViT-S"
        plt.text(x, y-0.5, text, ha='center', va='bottom', fontsize=14)

        plt.plot([0.7, 0.725], [data["acc1"][3], 78], linewidth=2, zorder=1000, color="black")

plt.title('Dynamic Inference Performance')
plt.xlabel(r'Patch Keep Rate $\rho$')
plt.ylabel('Validation Acc. (%)')
# plt.ylim(0.6, 1.0)
plt.ylim((70, 81))
# plt.xlim((0.375, 1.025))
plt.gca().invert_xaxis()
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('../figures/dynamicvit-comp.pdf', dpi=300, bbox_inches='tight')
plt.clf()
