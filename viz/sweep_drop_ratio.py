import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()

SECS_TO_HOUR = 3600

drop_ratios = [0, 10, 20, 30, 40, 50, 60, 70, 80]
runs = ["baseline", "magnitude_cyclic_80_0"]
accs = {
    'baseline': [83.598, 83.188, 82.656, 81.64, 79.972, 76.962, 72.58, 64.984, 49.872],
    'magnitude_cyclic_80_0': [83.354, 83.184, 82.92, 82.326, 81.556, 80.408, 78.322, 75.264, 69.064],
}

for run in runs:
    plt.plot(drop_ratios, accs[run], '-o', linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Patch Drop (%)')
plt.ylabel('Validation Accuracy')
plt.legend(loc=0)
plt.savefig('../figures/sweep-drop-ratio.pdf', dpi=300, bbox_inches='tight')
plt.clf()
