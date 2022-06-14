import os
import matplotlib
import matplotlib.pyplot as plt
SMALL_SIZE = 16
TICK_SIZE = 19
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=TICK_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=TICK_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure titl

from utils import load_jsonl

def ema(x, alpha=0.15):
    if len(x) == 0:
        return x
    
    y = [x[0]]
    for i in range(1, len(x)):
        y.append(alpha * y[-1] + (1 - alpha) * x[i])
    
    return y



ROOT = "/data/runs/simmim/simmim_finetune/"

runs = ['06-11-22', '06-12-22']

for run in runs:
    train_path = os.path.join(ROOT, run, "train_log.json")
    train_data = load_jsonl(train_path)
    plt.plot(ema(train_data['loss']), '-', linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Training Iteration')
plt.ylabel('Training Loss')
plt.legend(loc=0)
plt.savefig('figures/training-loss.pdf', dpi=300, bbox_inches='tight')
plt.clf()


for run in runs:
    val_path = os.path.join(ROOT, run, "val_log.json")
    val_data = load_jsonl(val_path)
    plt.plot(val_data['acc1'], '-', linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend(loc=0)
plt.savefig('figures/validation-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()
