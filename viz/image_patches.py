import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True,
                    help="root model/output dir")
args = parser.parse_args()

path = "pss/simmim/vit-b/magnitude_cyclic_80_0/image_patches_single.json"

with open(os.path.join(args.data, path)) as f:
    data = json.load(f)
    data["images"] = np.array(data["images"])


idx_to_class = {}
for key, value in data["class_to_idx"].items():
    idx_to_class[value] = key

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


        kept_patch_idxs = np.array(data["kept_patches"][j][0])
        kept_patches = np.array(data["kept_patches"][j][1])
        image_patch_idxs = kept_patch_idxs == i

        # handling class token (by omitting and incrementing to fix spatial position)
        image_patches = kept_patches[image_patch_idxs][1:]
        image_patches -= 1

        xy_idxs = get_xy_from_idxs(image_patches, image.shape[1] // 16)
        for x in range(image.shape[0] // 16):
            for y in range(image.shape[1] // 16):
                # set image area to white with black outline and alpha=0.5
                if (x, y) not in xy_idxs:
                    xs, xe = x*16, (x+1)*16
                    ys, ye = y*16, (y+1)*16
                    # blend image region with white to make transparent
                    image[ys:ye, xs:xe] = image[ys:ye, xs:xe] * 0.2 + 0.8

                    # add black border
                    image[ys:ys+1, xs:xe] = 0
                    image[ye-1:ye, xs:xe] = 0
                    image[ys:ye, xs:xs+1] = 0
                    image[ys:ye, xe-1:xe] = 0



        axs[i][j].imshow(image, interpolation="nearest")
        axs[i][j].axis("off")

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
