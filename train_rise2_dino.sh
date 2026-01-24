#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
torchrun --master_addr 127.0.0.1 --master_port 23334 --nproc_per_node 4 --nnodes 1 --node_rank 0 train.py --data_path /data/haoxiang/realdata_rise2_ready --ckpt_dir /data/haoxiang/logs/rise2_2601 --config configs/single_rise1.yaml

