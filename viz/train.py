import os
from utils import load_jsonl, ema, init_mpl
plt = init_mpl()


ROOT = "../runs/"
runs = ["base_lr_1.25e-3", "base_lr_6.25e-4", "pretrain_lr_1.25e-3"]
for run in runs:
    train_path = os.path.join(ROOT, run, "train_log.json")
    train_data = load_jsonl(train_path)
    plt.plot(ema(train_data['loss']), '-', linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Training Iteration')
plt.ylabel('Training Loss')
plt.legend(loc=0)
plt.savefig('../figures/training-loss.pdf', dpi=300, bbox_inches='tight')
plt.clf()


for run in runs:
    val_path = os.path.join(ROOT, run, "val_log.json")
    val_data = load_jsonl(val_path)
    plt.plot(val_data['acc1'], '-', linewidth=2, label=run)

plt.title('ImageNet (VIT)')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend(loc=0)
plt.savefig('../figures/validation-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()
