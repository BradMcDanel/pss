import os
import numpy as np

from utils import load_jsonl, ema, init_mpl
plt = init_mpl()


SECS_TO_HOUR = 3600

ROOT = "/data/runs/fracpatch/finetune/vit-b"
runs = ["baseline", "magnitude_cyclic_80_0"]
names = ["ViT-B", "+PSS"]
colors = ["#1F78B4", "#E31A1C"]
hashes = ['o', 's']
train_datas, val_datas, sweep_datas = {}, {}, {}
for run in runs:
    train_datas[run] = load_jsonl(os.path.join(ROOT, run, "train_log.json"))
    val_datas[run] = load_jsonl(os.path.join(ROOT, run, "val_log.json"))
    sweep_datas[run] = load_jsonl(os.path.join(ROOT, run, "sweep_drop.json"))


# two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

for i, run in enumerate(runs):
    total_time = np.cumsum(train_datas[run]["time"]) / SECS_TO_HOUR
    idxs = np.rint(np.linspace(0, len(total_time)-1, len(val_datas[run]["acc1"]))).astype('int')
    epoch_times = [total_time[i] for i in idxs]
    axs[0].plot(epoch_times, val_datas[run]["acc1"], "-", linewidth=2, label=names[i],
                color=colors[i])

    # add text and line for final accuracy
    y_pos = val_datas[run]["acc1"][-1]
    x_pos = epoch_times[-1]

    # add a hash marker for the max accuracy
    axs[0].plot(x_pos, y_pos, hashes[i], color=colors[i], markersize=6)

    # "acc% in x hours"
    if i == 0:
        axs[0].text(x_pos-4, y_pos-5, "%.2f%% in\n%.2f hours" % (y_pos, x_pos),
                    fontsize=12, color=colors[i], ha="center", va="center")
        axs[0].plot([x_pos, x_pos-4], [y_pos, y_pos-3], "k-", linewidth=1)

    else:
        axs[0].text(x_pos-3, y_pos-5, "%.2f%% in\n%.2f hours" % (y_pos, x_pos),
                    fontsize=12, color=colors[i], ha="center", va="center")
        axs[0].plot([x_pos, x_pos-3], [y_pos, y_pos-3], "k-", linewidth=1)
    



# plt.title('ImageNet (ViT-B)')
axs[0].set_title('Faster Training')
axs[0].set_xlabel('Training Time (Hours)')
axs[0].set_ylabel('Validation Acc. (%)')
axs[0].set_ylim((65, 85))
axs[0].legend(loc=0)

for i, run in enumerate(runs):
    drop_ratios = sweep_datas[run]["drop_ratio"]
    keep_ratios = [1 - drop_ratio for drop_ratio in drop_ratios]

    accs = sweep_datas[run]["acc1"]
    axs[1].plot(keep_ratios, accs, '-', linewidth=2, label=names[i], color=colors[i], marker=hashes[i])

    if i == 1:
        throughputs = sweep_datas[run]["throughput"]
        # add text of each throughput point
        for j, throughput in enumerate(throughputs):
            # round to nearest int
            throughput = int(throughput)
            # check y-axis range
            if accs[j] < 65 or  accs[j] > 84:
                continue

            if j == 0:
                x_pos = keep_ratios[j] - 0.06
                y_pos = accs[j] - 4
                axs[1].text(x_pos, y_pos, f"{throughput}\nimgs/sec", fontsize=10, ha='center')
                axs[1].plot([keep_ratios[j], x_pos + 0.02], [accs[j], y_pos + 2.5], color="black", linewidth=1)
            elif j == 4:
                x_pos = keep_ratios[j] - 0.04
                y_pos = accs[j] + 0.9
                axs[1].text(x_pos, y_pos, f"{throughput}\nimgs/sec", fontsize=10, ha='center')
                axs[1].plot([keep_ratios[j], x_pos], [accs[j], y_pos - 0.3], color="black", linewidth=1)
            elif j == 8:
                x_pos = keep_ratios[j] + 0.05
                y_pos = accs[j] + 7.5
                axs[1].text(x_pos, y_pos, f"{throughput}\nimgs/sec", fontsize=10, ha='center')
                axs[1].plot([keep_ratios[j], x_pos], [accs[j], y_pos - 0.3], color="black", linewidth=1)

            # add red line connecting throughput point to acc. point


axs[1].invert_xaxis()
axs[1].set_title('Dynamic Inference')
axs[1].set_xlabel(r'Patch Keep Rate $\rho$')
axs[1].set_ylim((65, 85))
axs[1].legend(loc=0)

plt.tight_layout()


plt.savefig('../figures/overview.pdf', dpi=300, bbox_inches='tight')
plt.clf()
