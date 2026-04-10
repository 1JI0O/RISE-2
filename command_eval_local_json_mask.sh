#!/bin/bash

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHONPATH="${SCRIPT_DIR}/airexo:${PYTHONPATH}" \
python "${SCRIPT_DIR}/eval_maskaware_json_260314.py" \
--type local \
--ckpt /data/haoxiang/logs/task0012-ca-pi-xie-mask-aware/policy_last.ckpt \
--calib_rise2 /data/haoxiang/data/task0012_260321/task0012_toys_basket_converted/calib/rise2_calib_20260104.npy \
--config "${SCRIPT_DIR}/configs/dual_ca_pi_xie_mask_aware_json.yaml" \
--host 127.0.0.1 \
--port 8000
