import os
import numpy as np
import json

from utils import init_mpl, load_imagenet_names
plt = init_mpl()
imagenet_names = load_imagenet_names()

def get_xy_from_idxs(idxs, img_width):
    xy_idxs = []
    for idx in idxs:
        x = idx % img_width
        y = idx // img_width
        xy_idxs.append((x, y))
    return xy_idxs

ROOT = "/data/runs/fracpatch/finetune/vit-b/magnitude_cyclic_80_0"
with open(os.path.join(ROOT, "image_patches.json")) as f:
    data = json.load(f)
    data["kept_patches"] = data["kept_patches"]
    data["images"] = np.array(data["images"])


idx_to_class = {}
for key, value in data["class_to_idx"].items():
    idx_to_class[value] = key

# plot 4 images horizontally
num_drop_ratios = len(data["drop_ratios"])
fig, axs = plt.subplots(4, num_drop_ratios, figsize=(20, 16), sharex=True)
for j in range(num_drop_ratios):
    for i in range(4):
        # transpose to get the image in the correct order
        image = data["images"][i].transpose(1, 2, 0)
        # map image to 0-1
        image = (image + abs(image.min())) / (image.max() - image.min())

        target = data["targets"][i]
        logits = np.array(data["logits"][j][i])
        logits = logits - logits.min()
        logits = logits / logits.sum()
        sorted_idxs = np.argsort(logits)[::-1]
        target_idx = np.where(sorted_idxs == target)[0][0]
        pred_idx = sorted_idxs[0]
        pred_conf = logits[pred_idx]

        axs[i][j].imshow(image, interpolation="nearest")
        axs[i][j].axis("off")


        kept_patch_idxs = np.array(data["kept_patches"][j][0])
        kept_patches = np.array(data["kept_patches"][j][1])
        image_patch_idxs = kept_patch_idxs == i

        # handling class token (by omitting and incrementing to fix spatial position)
        image_patches = kept_patches[image_patch_idxs][1:]
        image_patches -= 1

        xy_idxs = get_xy_from_idxs(image_patches, image.shape[1] // 16)
        for x in range(image.shape[0] // 16):
            for y in range(image.shape[1] // 16):
                # check if xy is in the list of kept patches
                if (x, y) not in xy_idxs:
                    # add frosted glass effect (trasparent white square over the patch)
                    axs[i][j].add_patch(plt.Rectangle((x*16, y*16), 16, 16, facecolor="white", edgecolor="black", alpha=0.8))

        if j == 0:
            class_name = imagenet_names[idx_to_class[data["targets"][i]]]
            # add text to left of image row (vertical)
            axs[i][j].text(-30, 120, class_name, fontsize=24, color="black", ha="left",
                           rotation=90, va="center")

        # add pred name and conf below image
        pred_class_name = imagenet_names[idx_to_class[pred_idx]]
        text = "{} ({:.2f}%)".format(pred_class_name, 100.*pred_conf)
        axs[i][j].text(110, image.shape[0] + 20, text, fontsize=24, color="black", 
                       ha="center", va="center")
                 
    # add title for column
    axs[0][j].set_title(r"$\rho={:1.1f}$".format(1 - data["drop_ratios"][j]), fontsize=26)

    
plt.tight_layout()
plt.subplots_adjust(wspace=-0.3)
plt.savefig("../figures/image-patches.pdf", dpi=300, bbox_inches="tight")
plt.clf()
