#!/bin/bash

set -euo pipefail

export PYTHONPATH="/home/haoxiang/rise2_mask_aware/airexo:${PYTHONPATH:-}"
export PYTHONPATH="/home/haoxiang/rise2_mask_aware/easyrobot:${PYTHONPATH}"

python eval_maskaware_single_tcp_anchor.py \
--type local \
--ckpt /data/haoxiang/logs/single_foar_purplebox_tcp_anchor/policy_last.ckpt \
--calib_rise2 /data/haoxiang/data/zihao_foar2/flip_0326_purple_box/calib/rise2_calib_single_foar_purplebox.npy \
--config configs/single_foar_purplebox_eval_tcp_anchor.yaml \
--host 127.0.0.1 \
--port 8000
