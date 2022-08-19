import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()

SECS_TO_HOUR = 3600

ROOT = "/data/runs/pss/finetune/vit-b"
runs = ["baseline", "magnitude_fixed_40",
        "magnitude_linear_80_0", "magnitude_cyclic_80_0"]
names = ["ViT-B", "+Fixed(0.6)", "+Linear(0.2, 1.0)", "+Cyclic(0.2, 1.0)"]
colors = ["#1F78B4", "#FF7F0F", "#33A02C", "#E31A1C"]
hashes = ['o', '^', 'v', 's']
plt.figure(figsize=(8, 4))
for i, run in enumerate(runs):
    train_path = os.path.join(ROOT, run, "sweep_drop.json")
    data = load_jsonl(train_path)
    keep_ratios = [1 - drop_ratio for drop_ratio in data["drop_ratio"]]
    plt.plot(keep_ratios, data["acc1"],
             label=names[i], color=colors[i], marker=hashes[i], linewidth=2)
    
    if i == 3:
        # Add throughput as text above each baseline point
        for j, keep_ratio in enumerate(keep_ratios):
            if j == 7:
                break
            if j == 6:
                x = keep_ratio + 0.01
                y = data["acc1"][j] - 0.01
            else:
                x = keep_ratio - 0.025
                y = data["acc1"][j] + 0.5
            throughput = data["throughput"][j]
            text = r"$\mathrm{%.2f}$" % throughput
            if j == 6:
                plt.text(x, y, text, ha='center', va='bottom', fontsize=12, rotation=-40)
            else:
                plt.text(x, y, text, ha='center', va='bottom', fontsize=12)

plt.title('PSS Dynamic Inference')
plt.xlabel(r'Patch Keep Rate $\rho$')
plt.ylabel('Validation Acc. (%)')
plt.ylim(0.6, 1.0)
plt.ylim((78, 84.5))
plt.xlim((0.375, 1.025))
plt.gca().invert_xaxis()
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('../figures/sweep-drop-ratio.pdf', dpi=300, bbox_inches='tight')
plt.clf()
