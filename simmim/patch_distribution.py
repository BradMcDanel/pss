# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import json
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from fvcore.nn import FlopCountAnalysis

from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from logger import create_logger
from utils import load_checkpoint, load_pretrained, reduce_tensor
from patch_scheduler import build_patch_scheduler

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained', type=str, help='path to pre-trained model')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument("--local_rank", type=int, default=0, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)
    config.defrost()
    config.DATA.NUM_WORKERS = 32
    config.freeze()
    print(config.DATA.NUM_WORKERS)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, logger, is_pretrain=False)

    if config.MODEL.TYPE == "vit":
        config.defrost()
        config.MODEL.TYPE = "fracpatch_vit"
        config.freeze()

    model = build_model(config, is_pretrain=False)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    load_pretrained(config, model_without_ddp, logger)

    model.module.set_patch_drop_func(config.TRAIN.PATCH_DROP_FUNC)
    drop_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
    results = {
        "drop_ratios": drop_ratios,
        "counts": [],
    }

    for drop_ratio in drop_ratios:
        model.module.set_patch_drop_ratio(drop_ratio)
        for idx, (images, _) in enumerate(data_loader_val):
            images = images.cuda(non_blocking=True)
            patch_info = model.module.get_patch_info(images)
            patch_idxs, idx_shape = patch_info
            patch_pos = patch_idxs[1]
            # remove cls token
            patch_pos = patch_pos[patch_pos != 0] - 1
            if drop_ratio == 0 and idx == 0:
                minlength = patch_pos.max().item() + 1
                
            if idx == 0:
                counts = torch.zeros(minlength, dtype=torch.int64, device="cpu")
            
            # bincount patch_pos and accumulate counts
            counts += torch.bincount(patch_pos.cpu(), minlength=minlength)
        
        results["counts"].append(counts.numpy().tolist())


    # get dir of args.resume
    save_path = os.path.join(config.OUTPUT, "patch_distribution.json")
    with open(save_path, 'w') as f:
        json.dump(results, f)


@torch.no_grad()
def validate(config, data_loader, model, epoch=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, target) in enumerate(data_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)

            # measure accuracy and record loss
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            acc1 = reduce_tensor(acc1)
            acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter.update(loss.item(), target.size(0))
            acc1_meter.update(acc1.item(), target.size(0))
            acc5_meter.update(acc5.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        # print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}_sweep_drop_ratio")

    main(config)
