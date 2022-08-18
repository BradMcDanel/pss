import os
import numpy as np
import json

from utils import init_mpl, load_imagenet_names
plt = init_mpl()
plt.rc("xtick", labelsize=10)  # fontsize of the tick labels
plt.rc("ytick", labelsize=10)  # fontsize of the tick labels
imagenet_names = load_imagenet_names()
NUM_IMAGES = 50000

def get_xy_from_idxs(idxs, img_width):
    xy_idxs = []
    for idx in idxs:
        x = idx % img_width
        y = idx // img_width
        xy_idxs.append((x, y))
    return xy_idxs

ROOT = "/data/runs/fracpatch/finetune/vit-b/magnitude_cyclic_80_0"
with open(os.path.join(ROOT, "patch_distribution.json")) as f:
    data = json.load(f)

num_drop_ratios = len(data["drop_ratios"]) - 1
fig, axs = plt.subplots(1, num_drop_ratios)

for j in range(num_drop_ratios):
    counts = data["counts"][j+1]
    counts = np.array(counts)

    w = int(np.sqrt(counts.shape[0]))

    # normalize counts
    counts = 100. * (counts / (NUM_IMAGES))

    counts = counts.reshape((w, w))

    axs[j].axis("off")

    # plot counts as 2d histogram
    axs[j].imshow(counts, cmap="hot", interpolation="nearest")

    # add colorbar next to the histogram
    vmin, vmax = counts.min(), counts.max()
    mid = (vmax + vmin) / 2.
    cbar = fig.colorbar(axs[j].images[-1], ax=axs[j], fraction=0.046,
                        pad=0.04)
    cbar.set_ticks([vmin, mid, vmax])
    cbar.set_ticklabels(['{:.0f}%'.format(vmin), '{:.0f}%'.format(mid), '{:.0f}%'.format(vmax)])

    # add title (drop ratio)
    axs[j].set_title(r"$\rho$={:.1f}".format(1 - data["drop_ratios"][j+1]), fontsize=10)

plt.tight_layout()
plt.subplots_adjust(wspace=0.5)
plt.savefig("../figures/patch-distribution.pdf", dpi=300, bbox_inches="tight")
plt.clf()
