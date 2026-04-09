#!/bin/bash

# 定时下载管理脚本（已合并到 download_data_retry.sh）
# 此脚本仅用于启动和管理下载任务

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_data_retry.sh"
PID_FILE="$SCRIPT_DIR/.download_script.pid"

# 检查是否已在运行
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    echo "下载任务已在运行 (PID: $(cat $PID_FILE))"
    echo "使用以下命令停止: kill $(cat $PID_FILE)"
    exit 1
fi

echo "启动定时下载任务..."
echo "- 立即开始下载"
echo "- 每天 18:00 自动重启"
echo "- 每天 08:30 自动停止"

# 启动下载脚本
nohup bash "$DOWNLOAD_SCRIPT" &
echo "已启动 (PID: $!)"
echo "日志文件: $SCRIPT_DIR/output.log"
echo "查看日志: tail -f $SCRIPT_DIR/output.log"
