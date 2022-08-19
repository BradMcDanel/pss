import math
import numpy as np

from utils import init_mpl
plt = init_mpl()

class LinearPatchScheduler:
    def __init__(self, start_patch_drop_ratio, end_patch_drop_ratio, 
                 total_epochs, iterations_per_epoch, cooldown_epochs=0,
                 curr_epoch=0):
        self.start_patch_drop_ratio = start_patch_drop_ratio
        self.end_patch_drop_ratio = end_patch_drop_ratio
        self.iterations_per_epoch = iterations_per_epoch
        self.total_epochs = total_epochs - cooldown_epochs
        self.cooldown_epochs = cooldown_epochs
        self.curr_iteration = 0 if curr_epoch == 0 else curr_epoch * iterations_per_epoch
        self.total_iterations = self.total_epochs * iterations_per_epoch


    def step(self):
        self.curr_iteration += 1


    def set_epoch(self, epoch):
        self.curr_iteration = epoch * self.iterations_per_epoch


    def get_patch_drop_ratio(self):
        iteration_pct = self.curr_iteration / self.total_iterations
        curr_patch_ratio = (1 - iteration_pct) * self.start_patch_drop_ratio + \
                           iteration_pct * self.end_patch_drop_ratio

        ratio = max(0.0, min(1.0, curr_patch_ratio))
        ratio = round(ratio * 5) / 5 # round to nearest 0.2
        return ratio



class CosinePatchScheduler:
    def __init__(self, start_patch_drop_ratio, end_patch_drop_ratio, total_epochs, 
                 iterations_per_epoch, cooldown_epochs=0, curr_epoch=0):
        self.start_patch_drop_ratio = start_patch_drop_ratio
        self.end_patch_drop_ratio = end_patch_drop_ratio
        self.total_epochs = total_epochs - cooldown_epochs
        self.cooldown_epochs = cooldown_epochs
        self.curr_iteration = 0 if curr_epoch == 0 else curr_epoch * iterations_per_epoch
        self.total_iterations = self.total_epochs * iterations_per_epoch


    def step(self):
        self.curr_iteration += 1


    def set_epoch(self, epoch):
        self.curr_iteration = epoch * self.iterations_per_epoch


    def get_patch_drop_ratio(self):
        if self.curr_iteration == 0:
            ratio = self.start_patch_drop_ratio
        elif self.curr_iteration >= self.total_iterations - 1:
            ratio = self.end_patch_drop_ratio
        else:
            ratio = (self.end_patch_drop_ratio + (self.start_patch_drop_ratio-self.end_patch_drop_ratio) *
                    (1+math.cos((self.curr_iteration)*math.pi / self.total_iterations)) / 2)

        ratio = max(0.0, min(1.0, ratio))
        ratio = round(ratio * 5) / 5 # round to nearest 0.2
        return ratio


class CyclicPatchScheduler:
    def __init__(self,start_patch_drop_ratio, end_patch_drop_ratio, total_epochs,
                 iterations_per_epoch=1200, cooldown_epochs=0, curr_epoch=0):
        self.start_patch_drop_ratio = start_patch_drop_ratio
        self.end_patch_drop_ratio = end_patch_drop_ratio
        self.curr_epoch = curr_epoch
        self.total_epochs = total_epochs - cooldown_epochs
        self.cooldown_epochs = cooldown_epochs
        self.iterations_per_epoch = iterations_per_epoch
        self.curr_iteration = 0 if curr_epoch == 0 else curr_epoch * iterations_per_epoch
        self.total_iterations = self.total_epochs * iterations_per_epoch

    def step(self):
        self.curr_iteration += 1


    def set_epoch(self, epoch):
        self.curr_iteration = epoch * self.iterations_per_epoch


    def get_patch_drop_ratio(self):
        if self.curr_iteration > self.total_iterations - 1:
            return 0.0

        curr_iteration  = self.curr_iteration % self.iterations_per_epoch
        if (curr_iteration <= (self.iterations_per_epoch // 2)):
            iterations_pct =  curr_iteration / ((self.iterations_per_epoch)/2)
        elif (curr_iteration > (self.iterations_per_epoch // 2)):
            iterations_pct =  (self.iterations_per_epoch - curr_iteration) / ((self.iterations_per_epoch)/2)

        curr_patch_ratio =  ((1 - iterations_pct) * self.start_patch_drop_ratio + \
                           iterations_pct * self.end_patch_drop_ratio)

        curr_patch_ratio = max(0.0, min(1.0, curr_patch_ratio))
        curr_patch_ratio = round(curr_patch_ratio * 5) / 5 # round to nearest 0.2
        return curr_patch_ratio


colors = ["#1F78B4", "#FF7F0F", "#33A02C", "#E31A1C"]
hashes = ['o', '^', 'v', 's']

ITERS = 2502
xs = []
patch_scheduler = LinearPatchScheduler(0.0, 0.0, 100, ITERS, 0)
baseline_ratios = []
for i in range(100*ITERS):
    curr = patch_scheduler.get_patch_drop_ratio()
    baseline_ratios.append(1 - curr)
    patch_scheduler.step()
    xs.append(i / 2502.0)

patch_scheduler = LinearPatchScheduler(0.4, 0.4, 100, ITERS, 0)
fixed_ratios = []
for i in range(100*ITERS):
    curr = patch_scheduler.get_patch_drop_ratio()
    fixed_ratios.append(1 - curr)
    patch_scheduler.step()

patch_scheduler = LinearPatchScheduler(0.8, -0.1, 100, ITERS, 0)
linear_ratios = []
for i in range(100*ITERS):
    curr = patch_scheduler.get_patch_drop_ratio()
    linear_ratios.append(1 - curr)
    patch_scheduler.step()


patch_scheduler = CyclicPatchScheduler(0.8, -0.1, 100, ITERS, 0)
cyclic_ratios = []
for i in range(100*ITERS):
    curr = patch_scheduler.get_patch_drop_ratio()
    cyclic_ratios.append(1 - curr)
    patch_scheduler.step()

fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

total_range = 100*ITERS
idxs = [int(i * total_range / 10.0) for i in range(10)]
idxs.append(total_range-1)
xs_hash = [xs[i] for i in idxs]
baseline_hash = [baseline_ratios[int(i)] for i in idxs]
fixed_hash = [fixed_ratios[int(i)] for i in idxs]
linear_hash = [linear_ratios[int(i)] for i in idxs]


# plot the ratios 
axs[0].plot(xs, baseline_ratios, color=colors[0], lw=2)
axs[0].plot(xs, fixed_ratios, color=colors[1], lw=2)
axs[0].plot(xs, linear_ratios, color=colors[2], lw=2)
axs[0].scatter(xs_hash, baseline_hash, marker=hashes[0], color=colors[0])
axs[0].scatter(xs_hash, fixed_hash, marker=hashes[1], color=colors[1])
axs[0].scatter(xs_hash, linear_hash, marker=hashes[2], color=colors[2])
axs[0].plot([], [], "-", label='Baseline', color=colors[0], marker=hashes[0])
axs[0].plot([], [], "-", label='Fixed(0.6)', color=colors[1], marker=hashes[1])
axs[0].plot([], [], "-", label='Linear(0.2, 1.0)', color=colors[2], marker=hashes[2])

axs[0].set_xlabel("Training Epochs")
axs[0].set_ylabel(r"Patch Keep Rate $\rho$")

# plot cyclic on axs[1]
cyclic_ratios = np.array(cyclic_ratios)
xs = np.array(xs)
cyclic_ratios = cyclic_ratios[:ITERS]
idxs = [int(i * ITERS / 10.0) for i in range(10)]
idxs.append(ITERS-1)
cyclic_hash = [cyclic_ratios[i] for i in idxs]
xs = np.arange(0, ITERS)
xs_hash = [xs[i] for i in idxs]
axs[1].plot(cyclic_ratios, color=colors[3], lw=2)
axs[1].scatter(xs_hash, cyclic_hash, marker=hashes[3], color=colors[3])
axs[1].plot([], [], "-", label='Cyclic(0.2, 1.0)', color=colors[3], marker=hashes[3], lw=2)
axs[1].set_xlabel("One Epoch (iters)")

# combine axs[0] and axs[1] legend and move legend to the right
lines_labels = [ax.get_legend_handles_labels() for ax in axs]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# increase line widths
for line in lines:
    line.set_linewidth(2)

fig.legend(lines, labels, loc="upper center",  bbox_to_anchor=(0.45, -0.10),
           ncol=4, handletextpad=0.1, handlelength=1.2, columnspacing=0.6)
# add title using text
fig.text(0.5, 0.95, "Patch Sampling Schedules", ha='center', fontsize=22)

plt.savefig('../figures/patch-schedules.pdf', bbox_inches='tight')
plt.clf()

