#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
torchrun --master_addr 127.0.0.1 --master_port 23335 --nproc_per_node 1 --nnodes 1 --node_rank 0 train.py --data_path /data/haoxiang/realdata_rise2_ready --ckpt_dir /data/haoxiang/logs/test --config configs/test.yaml

