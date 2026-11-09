#!/bin/bash
# ============================================================================
# runner：跑 parent 编排进程；崩溃则 60s 后重启（最多 9999 次）。
# 退出码 0 = 全部完成；99 = 操作者主动停止（不重启）。
# 由 start_download.sh 以 setsid 新会话拉起，脱离终端/harness。
#
# 用法（一般不直接跑）: bash run_download.sh --feature egomotion [其他参数]
# ============================================================================
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/xingao/code/NVlabs-alpamayo/physical_ai_av/.venv/bin/python
PARENT="$SCRIPT_DIR/download_features.py"
cd "$SCRIPT_DIR" || exit 1

try=0
PARENT_PID=""
STOP=0
trap 'STOP=1; [ -n "$PARENT_PID" ] && kill -TERM "$PARENT_PID" 2>/dev/null' TERM INT

while [ "$try" -lt 9999 ]; do
    "$PY" "$PARENT" "$@" &
    PARENT_PID=$!
    wait "$PARENT_PID"
    rc=$?
    PARENT_PID=""
    if [ "$rc" -eq 0 ]; then
        echo "[runner] ✔ 下载全部完成"
        exit 0
    fi
    if [ "$rc" -eq 99 ] || [ "$STOP" = 1 ]; then
        echo "[runner] 收到停止信号，退出"
        exit 0
    fi
    try=$((try + 1))
    echo "[runner] parent 退出码 $rc，60s 后重启（第 $try 次重启）"
    sleep 60 &
    wait $!
done
echo "[runner] ❌ 达到最大重启次数"
exit 1
