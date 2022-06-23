import torch
import math

def build_patch_scheduler(config):
    if config.TRAIN.PATCH_SCHEDULER.NAME == "linear":
        return LinearPatchScheduler(
            config.TRAIN.PATCH_SCHEDULER.START_PATCH_DROP_RATIO,
            config.TRAIN.PATCH_SCHEDULER.END_PATCH_DROP_RATIO,
            config.TRAIN.EPOCHS,
        )
    elif config.TRAIN.PATCH_SCHEDULER.NAME == "cosine":
        return CosinePatchScheduler(
            config.TRAIN.PATCH_SCHEDULER.START_PATCH_DROP_RATIO,
            config.TRAIN.PATCH_SCHEDULER.END_PATCH_DROP_RATIO,
            config.TRAIN.EPOCHS,
        )
    else:
        raise ValueError("Unknown patch scheduler: {}".format(config.TRAIN.PATCH_SCHEDULER.NAME))


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

class CyclicScheduler:
    #total number of iterations per epoch = 1200
    #total number of iterations = 120000
    def __init__(self,start_patch_drop_ratio, end_patch_drop_ratio, total_epochs,
                    iterations_per_epoch=1200,curr_epoch=0, curr_iterations = 0):
        self.start_patch_drop_ratio = start_patch_drop_ratio
        self.end_patch_drop_ratio = end_patch_drop_ratio
        self.total_epochs = total_epochs
        self.curr_epoch = curr_epoch
        self.iterations_per_epoch = iterations_per_epoch
        self.curr_iterations = curr_iterations
    #epoch step never actually called
    def epoch_step(self):
        self.curr_epoch += 1

    def iterations_step(self):
        if (self.curr_iterations < 1200):
            self.curr_iterations += 1
        elif (self.curr_iterations == 1200):
            self.curr_iterations = 0
            self.curr_epoch += 1

    def get_patch_drop_ratio(self):
        iterations_pct = self.curr_iterations / (self.iterations_per_epoch-1)
        print("iteration pct: ", iterations_pct)
        if (iterations_pct == 0.5):
            return 1.0
        elif (iterations_pct < 0.5):
            curr_patch_ratio = (1 - iterations_pct) * self.start_patch_drop_ratio + \
                           iterations_pct * self.end_patch_drop_ratio
            return curr_patch_ratio
        elif (iterations_pct > 0.5):
            curr_patch_ratio = (1 - iterations_pct) * self.end_patch_drop_ratio + \
                           iterations_pct * self.start_patch_drop_ratio
            return curr_patch_ratio

if __name__ == '__main__':
    patch_scheduler = CosinePatchScheduler(0.8, 0.0, 100)

    for i in range(100):
        print(patch_scheduler.get_patch_drop_ratio())
        patch_scheduler.step()
