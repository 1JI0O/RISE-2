#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/airexo:${SCRIPT_DIR}/easyrobot:${PYTHONPATH:-}"

conda run --no-capture-output -n rise2 python "${SCRIPT_DIR}/eval_original_save.py" \
  --type local \
  --ckpt /data/haoxiang/logs/single_foar_purplebox_original/policy_last.ckpt \
  --calib /data/haoxiang/data/zihao_foar2/flip_0326_purple_box/calib/rise2_calib_single_foar_purplebox.npy \
  --config "${SCRIPT_DIR}/configs/single_foar_purplebox_original.yaml" \
  --save-root "${SCRIPT_DIR}/deploy_capture_original"
