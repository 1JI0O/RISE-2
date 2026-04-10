#!/bin/bash

set -euo pipefail

export PYTHONPATH="/home/haoxiang/rise2_mask_aware/airexo:${PYTHONPATH:-}"

python eval_original.py \
--type local \
--ckpt /data/haoxiang/logs/single_foar_purplebox_original/policy_last.ckpt \
--calib /data/haoxiang/data/zihao_foar2/flip_0326_purple_box/calib/rise2_calib_single_foar_purplebox.npy \
--config configs/single_foar_purplebox_original.yaml \
--host 127.0.0.1 \
--port 8000
