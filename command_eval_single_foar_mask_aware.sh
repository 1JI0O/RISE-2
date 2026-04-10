#!/bin/bash

set -euo pipefail

export PYTHONPATH="/home/haoxiang/rise2_mask_aware/airexo:${PYTHONPATH:-}"

python eval_maskaware_json_260314.py \
--type local \
--ckpt /data/haoxiang/logs/single_foar_purplebox_mask_aware/policy_last.ckpt \
--calib_rise2 /data/haoxiang/data/zihao_foar2/flip_0326_purple_box/calib/rise2_calib_single_foar_purplebox.npy \
--config configs/single_foar_purplebox_eval_json.yaml \
--host 127.0.0.1 \
--port 8000
