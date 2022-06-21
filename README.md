
To run the fine-tuning script with a cosine patch rate, do: `python -m torch.distributed.launch --nproc_per_node 4 main_finetune.py --cfg configs/fracpatch_vit/random_cosine_80_0.yaml --data-path /data/datasets/imagenet --pretrained /data/models/simmim/simmim_pretrain__vit_base__img224__800ep.pth --batch-size 256 --output /data/runs/simmim/`
