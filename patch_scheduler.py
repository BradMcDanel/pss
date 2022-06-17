import torch
import math

def build_patch_scheduler(config):
    if config.TRAIN.PATCH_SCHEDULER.NAME == "linear":
        return LinearPatchScheduler(
            config.TRAIN.PATCH_SCHEDULER.START_PATCH_DROP_RATIO,
            config.TRAIN.PATCH_SCHEDULER.END_PATCH_DROP_RATIO,
            config.TRAIN.EPOCHS,
        )


class LinearPatchScheduler:
    def __init__(self, start_patch_drop_ratio, end_patch_drop_ratio, total_epochs, curr_epoch=0):
        self.start_patch_drop_ratio = start_patch_drop_ratio
        self.end_patch_drop_ratio = end_patch_drop_ratio
        self.total_epochs = total_epochs
        self.curr_epoch = curr_epoch

    def step(self):
        self.curr_epoch += 1

    def get_patch_drop_ratio(self):
        epoch_pct = self.curr_epoch / (self.total_epochs - 1)
        curr_patch_ratio = (1 - epoch_pct) * self.start_patch_drop_ratio + \
                           epoch_pct * self.end_patch_drop_ratio

        return max(0.0, min(1.0, curr_patch_ratio))

class CosinePatchScheduler:
    def __init__(self, start_patch_drop_ratio, end_patch_drop_ratio, total_epochs, curr_epoch = 0):
        self.start_patch_drop_ratio = start_patch_drop_ratio
        self.end_patch_drop_ratio = end_patch_drop_ratio
        self.total_epochs = total_epochs
        self.curr_epoch = curr_epoch

    def step(self):
        self.curr_epoch += 1

    def get_patch_drop_ratio(self):
        if self.curr_epoch == 0:
            return self.start_patch_drop_ratio
        elif self.curr_epoch == self.total_epochs - 1:
            return self.end_patch_drop_ratio
        elif self.curr_epoch > 0:
            return (self.end_patch_drop_ratio + (self.start_patch_drop_ratio-self.end_patch_drop_ratio) *
                    (1+math.cos((self.curr_epoch)*math.pi / self.total_epochs)) / 2)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    patch_scheduler = LinearPatchScheduler(0.1, 1.0, 100,)
    cosine_scheduler = CosinePatchScheduler(0.1, 1.0, 100,)

    y_axis = []
    x_axis = []
    for i in range(100):
        x = i
        y = cosine_scheduler.get_patch_drop_ratio()
        y_axis.append(y)
        x_axis.append(x)
        print(y)
        cosine_scheduler.step()
    #plt.plot(x_axis, y_axis)
    #plt.show()