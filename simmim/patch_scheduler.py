import math

def build_patch_scheduler(config, iterations_per_epoch):
    if config.TRAIN.PATCH_SCHEDULER.NAME == "linear":
        return LinearPatchScheduler(
            config.TRAIN.PATCH_SCHEDULER.START_PATCH_DROP_RATIO,
            config.TRAIN.PATCH_SCHEDULER.END_PATCH_DROP_RATIO,
            config.TRAIN.EPOCHS,
            iterations_per_epoch,
            config.TRAIN.PATCH_SCHEDULER.COOLDOWN_EPOCHS,
        )
    elif config.TRAIN.PATCH_SCHEDULER.NAME == "cosine":
        return CosinePatchScheduler(
            config.TRAIN.PATCH_SCHEDULER.START_PATCH_DROP_RATIO,
            config.TRAIN.PATCH_SCHEDULER.END_PATCH_DROP_RATIO,
            config.TRAIN.EPOCHS,
            iterations_per_epoch,
            config.TRAIN.PATCH_SCHEDULER.COOLDOWN_EPOCHS,
        )
    elif config.TRAIN.PATCH_SCHEDULER.NAME == "cyclic":
        return CyclicPatchScheduler(
            config.TRAIN.PATCH_SCHEDULER.START_PATCH_DROP_RATIO,
            config.TRAIN.PATCH_SCHEDULER.END_PATCH_DROP_RATIO,
            config.TRAIN.EPOCHS,
            iterations_per_epoch,
            config.TRAIN.PATCH_SCHEDULER.COOLDOWN_EPOCHS,
        )
    else:
        raise ValueError("Unknown patch scheduler: {}".format(config.TRAIN.PATCH_SCHEDULER.NAME))


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


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    patch_scheduler = LinearPatchScheduler(0.8, -0.1, 100, 1200, 0)
    linear_ratios = []
    for i in range(100*1200):
        curr = patch_scheduler.get_patch_drop_ratio()
        linear_ratios.append(curr)
        patch_scheduler.step()

    patch_scheduler = CosinePatchScheduler(0.8, -0.1, 100, 1200, 0)
    cosine_ratios = []
    for i in range(100*1200):
        curr = patch_scheduler.get_patch_drop_ratio()
        cosine_ratios.append(curr)
        patch_scheduler.step()

    patch_scheduler = CyclicPatchScheduler(0.8, -0.1, 100, 1200, 0)
    cyclic_ratios = []
    for i in range(100*1200):
        curr = patch_scheduler.get_patch_drop_ratio()
        cyclic_ratios.append(curr)
        patch_scheduler.step()

    plt.plot(linear_ratios, label='linear')
    plt.plot(cosine_ratios, label='cosine')
    plt.plot(cyclic_ratios, label='cyclic')
    plt.legend()
    plt.savefig("patch_scheduler.png")
    plt.close()

    # compute mean of each scheduler
    linear_mean = sum(linear_ratios) / len(linear_ratios)
    cosine_mean = sum(cosine_ratios) / len(cosine_ratios)
    cyclic_mean = sum(cyclic_ratios) / len(cyclic_ratios)

    print("linear mean: {}".format(linear_mean))
    print("cosine mean: {}".format(cosine_mean))
    print("cyclic mean: {}".format(cyclic_mean))
