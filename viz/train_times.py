import argparse
import os

import numpy as np

from utils import load_jsonl, init_mpl
plt = init_mpl()

SECS_TO_HOUR = 3600
ITERS_PER_BATCH = 2502

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True,
                    help="root model/output dir")
args = parser.parse_args()

runs = ["pss/simmim/vit-b/baseline", "pss/simmim/vit-b/magnitude_cyclic_80_0"]
names = ["ViT-B", "+Cyclic(0.2, 1.0)"]
colors = ["#1F78B4", "#E31A1C"]
plt.figure(figsize=(8, 4))
for i, run in enumerate(runs):
    train_path = os.path.join(args.data, run, "train_log.json")
    data = load_jsonl(train_path)
    # take second epoch (to avoid warm-up timing)
    times = data["time"][ITERS_PER_BATCH:2*ITERS_PER_BATCH]
    patch_keep_ratios = data["patch_drop_ratio"][ITERS_PER_BATCH:2*ITERS_PER_BATCH]
    patch_keep_ratios = [1 - p for p in patch_keep_ratios]

    # convert to ms
    times = [1000 * t for t in times]

    # remove large outliers with numpy
    times = np.array(times)
    patch_keep_ratios = np.array(patch_keep_ratios)
    idxs = np.abs(times - np.mean(times)) < 3 * np.std(times)
    times = times[idxs]
    patch_keep_ratios = patch_keep_ratios[idxs]

    plt.plot(times, '-', linewidth=2, label=names[i], color=colors[i])

    # find indexs where patch_keep_ratio changes
    if i == 1:
        idxs = np.where(np.diff(patch_keep_ratios) != 0)[0]
        idxs = np.append(idxs, len(patch_keep_ratios) - 1)
        # compute average time per patch_keep_ratio
        times_per_patch_keep_ratio = []
        for j, idx in enumerate(idxs):
            if j == 0:
                times_per_patch_keep_ratio.append(np.mean(times[:idx]))
            else:
                times_per_patch_keep_ratio.append(np.mean(times[idxs[j-1]:idx]))
            
        # plot a horizontal line at each patch_keep_ratio for the average time
        for j, idx in enumerate(idxs):
            t = times_per_patch_keep_ratio[j]
            if j == 0:
                plt.plot([0, idx], [t, t], '--', color='k', linewidth=2)
            else:
                plt.plot([idxs[j-1], idx], [t, t], '--', color='k', linewidth=2)
            
            text = r"$\rho$={:.1f}".format(patch_keep_ratios[idx])
            if j == 0:
                plt.text(idx+320, t-15, text, ha='right', va='bottom')
            elif j < 4:
                plt.text(idx+320, t-10, text, ha='right', va='bottom')
            elif j == 4:
                plt.text(idx-130, t+10, text, ha='right', va='bottom')

plt.title('Single Epoch Training Time')
plt.xlabel('Iteration Index (single epoch)')
plt.ylabel('Iteration Time (ms)')

plt.legend(loc="lower right", bbox_to_anchor=(0.8, 0.0))
plt.tight_layout()
plt.savefig('../figures/train-times.pdf', dpi=300, bbox_inches='tight')
plt.clf()
