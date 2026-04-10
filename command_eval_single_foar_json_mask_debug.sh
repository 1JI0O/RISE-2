#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHONPATH="${SCRIPT_DIR}/airexo:${PYTHONPATH:-}" \
conda run --no-capture-output -n rise2 python "${SCRIPT_DIR}/eval_maskaware_json_single_debug.py" \
--type local \
--ckpt /data/haoxiang/logs/single_foar_purplebox_mask_aware/policy_last.ckpt \
--calib_rise2 /data/haoxiang/data/zihao_foar2/flip_0326_purple_box/calib/rise2_calib_single_foar_purplebox.npy \
--config "${SCRIPT_DIR}/configs/single_foar_purplebox_eval_json.yaml" \
--host 127.0.0.1 \
--port 8000 \
--debug-scene-dir /data/haoxiang/data/zihao_foar2/flip_0326_purple_box/train/scene_0001 \
--debug-frame-id 1774521917807 \
--debug-h5 /data/haoxiang/data/zihao_foar2/flip_0326_purple_box/train/scene_0001/lowdim/lowdim.h5 \
--debug-left-suffix 062770
