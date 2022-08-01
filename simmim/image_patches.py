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

    model = build_model(config, is_pretrain=False)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    load_pretrained(config, model_without_ddp, logger)

    model.module.set_patch_drop_func("magnitude")
    print(model.module.patch_drop_func)
    drop_ratios = [0.0, 0.2, 0.4, 0.6, 0.8]
    results = {
        "drop_ratios": drop_ratios,
        "images": None,
        "kept_patches": [],
        "logits": [],
        "class_to_idx": dataset_train.class_to_idx,
    }
    # empty tensor
    all_images, all_targets = [], []
    for idx, (images, target) in enumerate(data_loader_val):
        if idx == 300:
            break
        all_images.extend(images)
        all_targets.extend(target)

    # shuffle images and sample 16
    torch.manual_seed(0)
    all_images = torch.stack(all_images)
    all_targets = torch.stack(all_targets)
    perm_idxs = torch.randperm(all_images.shape[0])
    all_images = all_images[perm_idxs]
    all_images = all_images[:4]
    all_targets = all_targets[perm_idxs]
    all_targets = all_targets[:4].tolist()
    images = all_images.cuda()

    results["images"] = images.cpu().numpy().tolist()
    results["targets"] = all_targets

    for drop_ratio in drop_ratios:
        model.module.set_patch_drop_ratio(drop_ratio)
        with torch.no_grad():
            patch_idxs, idx_shape = model.module.get_patch_info(images)
            results["kept_patches"].append((patch_idxs[0].cpu().numpy().tolist(), patch_idxs[1].cpu().numpy().tolist()))
           
            logits = model(images)
            logits = logits.cpu().numpy().tolist()
            results["logits"].append(logits)


    # get dir of args.resume
    save_path = os.path.join(config.OUTPUT, "image_patches.json")

    with open(save_path, 'w') as f:
        json.dump(results, f)


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
