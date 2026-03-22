#!/usr/bin/env bash
set -euo pipefail

# 用于在 SAM2 服务器机器上排查“能 ping 通但端口不通”
# 用法：
#   bash diag_sam2_server.sh
#   PORT=8976 SERVER_IP=58.34.245.178 bash diag_sam2_server.sh

PORT="${PORT:-8976}"
SERVER_IP="${SERVER_IP:-}"

echo "[diag-server] ===== basic info ====="
echo "[diag-server] host=$(hostname)"
echo "[diag-server] ips=$(hostname -I 2>/dev/null || true)"
echo "[diag-server] port=${PORT}"
if [[ -n "${SERVER_IP}" ]]; then
  echo "[diag-server] server_ip=${SERVER_IP}"
fi

echo

echo "[diag-server] ===== listening sockets (ss) ====="
if command -v ss >/dev/null 2>&1; then
  ss -lntp | grep ":${PORT}" || echo "[diag-server] no listening tcp socket on :${PORT}"
else
  echo "[diag-server] ss not found"
fi

echo

echo "[diag-server] ===== local tcp connect test ====="
python - "${PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
for host in ["127.0.0.1", "0.0.0.0"]:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect((host, port))
        print(f"[diag-server] local tcp ok: {host}:{port}")
    except Exception as e:
        print(f"[diag-server] local tcp fail: {host}:{port} ({type(e).__name__}: {e})")
    finally:
        s.close()
PY

echo

echo "[diag-server] ===== bind ip tcp connect test (optional) ====="
if [[ -n "${SERVER_IP}" ]]; then
python - "${SERVER_IP}" "${PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2.0)
try:
    s.connect((host, port))
    print(f"[diag-server] bind-ip tcp ok: {host}:{port}")
except Exception as e:
    print(f"[diag-server] bind-ip tcp fail: {host}:{port} ({type(e).__name__}: {e})")
finally:
    s.close()
PY
else
  echo "[diag-server] skip (set SERVER_IP to test, e.g. SERVER_IP=58.34.245.178)"
fi

echo

echo "[diag-server] ===== firewall quick check ====="
if command -v ufw >/dev/null 2>&1; then
  ufw status || true
else
  echo "[diag-server] ufw not installed"
fi

if command -v iptables >/dev/null 2>&1; then
  iptables -S | sed -n '1,60p' || true
else
  echo "[diag-server] iptables not found"
fi

echo

echo "[diag-server] done"
