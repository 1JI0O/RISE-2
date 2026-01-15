#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,3
torchrun --master_addr 127.0.0.1 --master_port 23334 --nproc_per_node 2 --nnodes 1 --node_rank 0 train.py --data_path /data/haoxiang/realdata_rise2_ready --ckpt_dir logs/single_rise1_run --config configs/single_rise1.yaml

