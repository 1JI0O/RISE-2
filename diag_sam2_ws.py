#!/usr/bin/env python3
"""诊断 SAM2 远程连接问题：TCP / WebSocket 握手 / 可选 infer。

用法示例：
  python diag_sam2_ws.py --host 58.34.245.178 --port 8976
  python diag_sam2_ws.py --host 58.34.245.178 --port 8976 --infer
"""

import argparse
import inspect
import socket
import sys
import traceback

import msgpack


def check_tcp(host, port, timeout):
    print(f"[diag] TCP connect {host}:{port} timeout={timeout}s")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        local = s.getsockname()
        print(f"[diag] TCP OK local={local[0]}:{local[1]}")
        return True
    except Exception as exc:
        print(f"[diag] TCP FAIL: {type(exc).__name__}: {exc}")
        return False
    finally:
        s.close()


def _build_ws_connect_kwargs(connect_fn, timeout):
    params = inspect.signature(connect_fn).parameters
    kwargs = {}

    if "compression" in params:
        kwargs["compression"] = None
    if "max_size" in params:
        kwargs["max_size"] = None
    if "open_timeout" in params:
        kwargs["open_timeout"] = timeout
    if "close_timeout" in params:
        kwargs["close_timeout"] = timeout
    if "proxy" in params:
        # 防止环境代理干扰本地/内网连接
        kwargs["proxy"] = None

    return kwargs


def check_ws_and_optional_infer(host, port, timeout, do_infer):
    print(f"[diag] WS connect ws://{host}:{port}")

    try:
        import websockets.sync.client
    except Exception as exc:
        print(f"[diag] WS FAIL: cannot import websockets.sync.client: {exc}")
        return False

    uri = f"ws://{host}:{port}"
    connect_fn = websockets.sync.client.connect
    kwargs = _build_ws_connect_kwargs(connect_fn, timeout)

    conn = None
    try:
        conn = connect_fn(uri, **kwargs)
        raw = conn.recv()
        ready = msgpack.unpackb(raw, raw=False)
        print(f"[diag] WS handshake payload: {ready}")

        if not isinstance(ready, dict) or ready.get("status") != "ready":
            print("[diag] WS FAIL: handshake payload is not {'status': 'ready'}")
            return False

        print("[diag] WS handshake OK")

        if do_infer:
            # 发送一个很小的伪造 color；服务端会 np.asarray 后处理
            color = [[[0, 0, 0] for _ in range(8)] for _ in range(8)]
            req = {"color": color}
            conn.send(msgpack.packb(req, use_bin_type=True))
            resp_raw = conn.recv()
            resp = msgpack.unpackb(resp_raw, raw=False)

            reason = resp.get("reason") if isinstance(resp, dict) else None
            mask = resp.get("mask") if isinstance(resp, dict) else None
            mask_type = type(mask).__name__
            print(f"[diag] infer response reason={reason} mask_type={mask_type}")
            print("[diag] infer round-trip OK (注意：首次可能在服务端弹标注窗口)")

        return True
    except Exception as exc:
        print(f"[diag] WS FAIL: {type(exc).__name__}: {exc}")
        print("[diag] traceback:")
        traceback.print_exc()
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Diagnose SAM2 remote connectivity")
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--infer", action="store_true", help="Also send one minimal infer request")
    args = parser.parse_args()

    ok_tcp = check_tcp(args.host, args.port, args.timeout)
    if not ok_tcp:
        sys.exit(2)

    ok_ws = check_ws_and_optional_infer(args.host, args.port, args.timeout, args.infer)
    if not ok_ws:
        sys.exit(3)

    print("[diag] ALL OK")


if __name__ == "__main__":
    main()
