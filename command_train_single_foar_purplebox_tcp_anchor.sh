#!/bin/bash

set -euo pipefail

export LIBGL_ALWAYS_SOFTWARE=1
export MPLBACKEND=Agg

DATA_PATH=/data/haoxiang/data/zihao_foar2/flip_0326_purple_box
CKPT_DIR=/data/haoxiang/logs/single_foar_purplebox_tcp_anchor
CONFIG=configs/single_foar_purplebox_tcp_anchor.yaml

# 单卡正式训练
# CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n rise2 torchrun \
# --master_addr 127.0.0.1 --master_port 23333 \
# --nproc_per_node 1 --nnodes 1 --node_rank 0 \
# train.py \
# --data_path ${DATA_PATH} \
# --ckpt_dir ${CKPT_DIR} \
# --config ${CONFIG}

# 四卡示例
CUDA_VISIBLE_DEVICES=4,5,6,7 conda run --no-capture-output -n rise2 torchrun \
--master_addr 127.0.0.1 --master_port 23333 \
--nproc_per_node 4 --nnodes 1 --node_rank 0 \
train_tcp_anchor.py \
--data_path ${DATA_PATH} \
--ckpt_dir ${CKPT_DIR} \
--config ${CONFIG}
