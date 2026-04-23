#!/bin/bash

# 下载启动脚本（包装脚本）
# 功能：
#   - 检查是否已有下载任务在运行
#   - 启动 download_scheduled.sh（包含定时和重试逻辑）
#   - 管理PID文件，防止重复启动
#
# 使用方式：
#   - 手动启动：bash start_download_scheduled.sh
#   - 通过at命令定时启动：at 17:30 -f start_download_scheduled.sh
#   - 停止任务：kill $(cat .download_scheduled.pid)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="$SCRIPT_DIR/download_scheduled.sh"
PID_FILE="$SCRIPT_DIR/.download_scheduled.pid"
LOG_FILE="$SCRIPT_DIR/download_scheduled.log"

# 检查是否已在运行
if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
    echo "❌ 下载任务已在运行 (PID: $(cat $PID_FILE))"
    echo "使用以下命令停止: kill $(cat $PID_FILE)"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 启动定时下载任务..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 功能说明："
echo "  ✓ 自动等待到 18:00 开始下载"
echo "  ✓ 每天 08:30 自动停止并进入等待"
echo "  ✓ 失败自动重试（最多9999次）"
echo "  ✓ 连续失败5次后自动清理缓存"
echo "  ✓ 自动检测416错误并修复"
echo "  ✓ 运行时段：18:00 - 次日 08:30"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 启动下载脚本并重定向日志
nohup bash "$DOWNLOAD_SCRIPT" > "$LOG_FILE" 2>&1 &
PID=$!
sleep 1  # 等待进程启动
echo "$PID" > "$PID_FILE"

echo ""
echo "✅ 已启动 (PID: $PID)"
echo "📄 日志文件: $LOG_FILE"
echo "👀 查看日志: tail -f $LOG_FILE"
echo "🛑 停止任务: kill $PID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
