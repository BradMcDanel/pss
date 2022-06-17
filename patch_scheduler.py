

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

if __name__ == '__main__':
    patch_scheduler = LinearPatchScheduler(0.1, 1.0, 100)

    for i in range(100):
        print(patch_scheduler.get_patch_drop_ratio())
        patch_scheduler.step()
