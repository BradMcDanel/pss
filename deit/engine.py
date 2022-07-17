# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import json
import time
import os

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, patch_schedule = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10


    cpu_time_start = time.time()
    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        lr = optimizer.param_groups[0]['lr']

        if patch_schedule is not None:
            model.module.set_patch_drop_ratio(patch_schedule.get_patch_drop_ratio())

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
                    
        with torch.cuda.amp.autocast():
            cpu_time_end = time.time()
            cpu_time = cpu_time_end - cpu_time_start
            gpu_time_start = time.time()
            outputs = model(samples)
            # loss = criterion(samples, outputs, targets)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)



        gpu_time_end = time.time()
        gpu_time = gpu_time_end - gpu_time_start
        total_time = gpu_time + cpu_time

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if patch_schedule is not None:
            patch_schedule.step()


        # if rank is 0, then print out the training loss
        if utils.get_rank() == 0:
            train_log_file = os.path.join(args.output_dir, 'train_log.json')
            train_info = {
                'epoch': epoch,
                'iteration': len(data_loader) * epoch + idx,
                'loss': loss_value,
                'lr': lr,
                'time': total_time,
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
            }

            if patch_schedule is not None:
                train_info['patch_drop_ratio'] = patch_schedule.get_patch_drop_ratio()

            with open(train_log_file, 'a') as f:
                f.write(json.dumps(train_info) + '\n')


        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        cpu_time_start = time.time()


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch=None, args=None):
    criterion = torch.nn.CrossEntropyLoss()


    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    start_time = time.time()
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    end_time = time.time()
    total_time = end_time - start_time

    if utils.get_rank() == 0 and epoch is not None:
        val_log_file = os.path.join(args.output_dir, 'val_log.json')
        val_info = {
            'epoch': epoch,
            'loss': metric_logger.loss.global_avg,
            'acc1': metric_logger.acc1.global_avg,
            'acc5': metric_logger.acc5.global_avg,
            'time': total_time,
        }
        with open(val_log_file, 'a') as f:
            f.write(json.dumps(val_info) + '\n')


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
