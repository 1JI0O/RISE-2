#!/usr/bin/env bash
set -euo pipefail

# 终端2（部署机 / rise2 环境）实机评估入口。
# 说明：
# - SAM2 分割服务在远端：58.34.245.178:8976
# - 策略推理默认本地（TYPE=local）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/airexo:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/dual_teleop_dino_sam2.yaml}"
TYPE="${TYPE:-local}"                                 # local | remote（策略推理）
HOST="${HOST:-127.0.0.1}"                             # 仅 TYPE=remote 时生效（策略服务地址）
PORT="${PORT:-8000}"                                  # 仅 TYPE=remote 时生效（策略服务端口）
SAM2_REMOTE_HOST="${SAM2_REMOTE_HOST:-58.34.245.178}" # SAM2 分割服务地址（跨机器默认）
SAM2_REMOTE_PORT="${SAM2_REMOTE_PORT:-8976}"          # SAM2 分割服务端口
CKPT="${CKPT:-/home/ubuntu/data/haoxiang/aware_0012_policy_last.ckpt}"
CALIB_AIREXO="${CALIB_AIREXO:-/home/ubuntu/data/haoxiang/1770288769108.npy}"
CALIB_RISE2="${CALIB_RISE2:-/home/ubuntu/data/haoxiang/rise2_calib_1770288769108.npy}"

echo "[eval_rise2] policy_type=${TYPE} policy_endpoint=${HOST}:${PORT} sam2_endpoint=${SAM2_REMOTE_HOST}:${SAM2_REMOTE_PORT}"

python - "${SAM2_REMOTE_HOST}" "${SAM2_REMOTE_PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3.0)
try:
    s.connect((host, port))
    print(f"[eval_rise2] sam2 tcp precheck ok: {host}:{port}")
except Exception as e:
    print(f"[eval_rise2] sam2 tcp precheck failed: {host}:{port} ({e})")
    sys.exit(2)
finally:
    s.close()
PY

SAM2_REMOTE_HOST="${SAM2_REMOTE_HOST}" SAM2_REMOTE_PORT="${SAM2_REMOTE_PORT}" \
python "${SCRIPT_DIR}/eval_sam2_mask_final.py" \
  --type "${TYPE}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --calib_airexo "${CALIB_AIREXO}" \
  --calib_rise2 "${CALIB_RISE2}"
