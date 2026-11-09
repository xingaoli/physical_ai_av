#!/bin/bash
# ============================================================================
# 通用 chunk 特征下载 启动/停止/查看（后台、脱离终端与 harness、防重复启动）
#
#   启动:  bash start_download.sh start --feature egomotion
#   查看:  bash start_download.sh status --feature egomotion
#   停止:  bash start_download.sh stop   --feature egomotion
#   日志:  tail -f download_egomotion.log   （本目录，按 feature 命名）
#
# 可选参数直接追加（透传给 download_features.py），例如：
#   bash start_download.sh start --feature egomotion --workers 8 --chunks 0-99
#
# 原理: setsid 新会话 + nohup + stdin=/dev/null，孤儿化(PPID->1)后台执行，
#       当前终端/harness 退出不影响。停止 = kill 整个进程组(runner+parent+child)。
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_download.sh"
DEFAULT_LOCAL_DIR=/mnt/hdd_data/public_data/PhysicalAI-Autonomous-Vehicles-all

# 从剩余参数里解析 --feature（status/stop/start 共用），并校验必填
ACTION="${1:-start}"; shift || true
FEATURE=""
ARGS=("$@")
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    if [ "${ARGS[$i]}" = "--feature" ] && [ $((i + 1)) -lt ${#ARGS[@]} ]; then
        FEATURE="${ARGS[$((i + 1))]}"
    fi
    i=$((i + 1))
done
if [ -z "$FEATURE" ] || [ "$ACTION" != "start" ] && [ "$ACTION" != "stop" ] && [ "$ACTION" != "status" ]; then
    echo "用法: bash $0 [start|stop|status] --feature <feature名> [其他透传参数]"
    echo "feature 名单见 $DEFAULT_LOCAL_DIR/features.csv（如 egomotion, camera_front_wide_120fov, camera_intrinsics）"
    exit 1
fi

PID_FILE="$SCRIPT_DIR/.download_${FEATURE}.pid"
LOG="$SCRIPT_DIR/download_${FEATURE}.log"

alive() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "$ACTION" in
start)
    if alive; then
        echo "❌ feature '$FEATURE' 已在运行 (PGID: $(cat "$PID_FILE"))。先停止: bash $0 stop --feature $FEATURE"
        exit 1
    fi
    rm -f "$PID_FILE"
    setsid nohup bash "$RUNNER" "${ARGS[@]}" </dev/null >>"$LOG" 2>&1 &
    PID=$!
    sleep 2
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "❌ 启动失败，最近日志："; tail -5 "$LOG"; exit 1
    fi
    echo "$PID" > "$PID_FILE"
    echo "✅ feature '$FEATURE' 已后台启动 (PGID/PID: $PID，已脱离当前终端)"
    echo "📄 日志:   tail -f $LOG"
    echo "🛑 停止:   bash $0 stop --feature $FEATURE"
    echo "👀 查看:   bash $0 status --feature $FEATURE"
    ;;
stop)
    if ! alive; then
        echo "feature '$FEATURE' 没有运行中的任务"; rm -f "$PID_FILE"; exit 0
    fi
    PGID=$(cat "$PID_FILE")
    kill -TERM -- -"$PGID" 2>/dev/null
    sleep 3
    kill -KILL -- -"$PGID" 2>/dev/null
    rm -f "$PID_FILE"
    echo "🛑 已停止 (PGID: $PGID)。已完成文件保留，重新 start 自动续传"
    ;;
status)
    if alive; then
        echo "▶️ feature '$FEATURE' 运行中 (PGID: $(cat "$PID_FILE"))"
    else
        echo "⏹️ feature '$FEATURE' 未运行"
    fi
    echo "--- 最近进度 ---"
    grep "📊" "$LOG" 2>/dev/null | tail -3
    echo "--- 日志最后 8 行 ---"
    tail -8 "$LOG" 2>/dev/null
    ;;
esac
