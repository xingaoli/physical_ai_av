#!/bin/bash

# ============================================================================
# 定时下载脚本（核心逻辑）
# ============================================================================
# 调用关系：
#   start_download_scheduled.sh（包装脚本，PID管理和nohup启动）
#     └─> download_scheduled.sh（本脚本，定时控制和重试逻辑）
#           └─> download_data.py（Python下载脚本）
#
# 功能：
#   - 严格限制运行时间：每天18:00开始下载，次日08:30停止
#   - 非下载时段（08:30-18:00）保持等待，不启动下载
#   - 下载失败自动重试（最多9999次）
#   - 连续失败5次后自动清理损坏的缓存
#   - 自动检测416错误并立即清理缓存
#   - 详细的日志记录（时间、状态、错误信息）
#
# 使用方式：
#   - 通过 start_download.sh 启动（推荐，可随时启动，自动等待到18:00）
#   - 或直接运行：bash download_scheduled.sh
#   - 运行时间：每天18:00-次日08:30，其他时间自动等待
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/../.."
MAX_RETRIES=9999
CONSECUTIVE_FAILURES_THRESHOLD=5  # 连续失败次数阈值，超过后清理缓存
SHOULD_STOP=0
DOWNLOAD_PID=""
FAILURE_COUNT_FILE="$SCRIPT_DIR/.consecutive_failures"  # 用于跨子shell记录连续失败次数

# 初始化失败计数文件
init_failure_count() {
    echo "0" > "$FAILURE_COUNT_FILE"
}

# 读取连续失败次数
get_failure_count() {
    if [ -f "$FAILURE_COUNT_FILE" ]; then
        cat "$FAILURE_COUNT_FILE"
    else
        echo "0"
    fi
}

# 设置连续失败次数
set_failure_count() {
    echo "$1" > "$FAILURE_COUNT_FILE"
}

# 增加连续失败次数
increment_failure_count() {
    local count=$(get_failure_count)
    count=$((count + 1))
    set_failure_count "$count"
    echo "$count"
}

# 重置连续失败次数
reset_failure_count() {
    set_failure_count "0"
    echo "0"
}

# 初始化
init_failure_count

# 强制停止所有下载相关进程
force_stop_download() {
    log "⚠️  开始强制停止所有下载相关进程..."
    
    # 方法1: 通过pkill强制终止Python下载进程
    local pids=$(pgrep -f "python.*download_data.py" 2>/dev/null)
    if [ ! -z "$pids" ]; then
        log "  📌 发现Python下载进程: $pids"
        echo "$pids" | while read pid; do
            kill -SIGKILL "$pid" 2>/dev/null
            log "    ✓ 已终止 PID: $pid"
        done
    fi
    
    # 方法2: 也尝试用pkill -f一次性清理
    pkill -9 -f "python.*download_data.py" 2>/dev/null
    
    # 方法3: 如果有子shell的run_download后台进程，也一并清理
    local bg_pids=$(pgrep -f "bash.*download_scheduled.sh" 2>/dev/null | grep -v "^$$\$")
    if [ ! -z "$bg_pids" ]; then
        log "  📌 发现相关bash进程: $bg_pids"
        echo "$bg_pids" | while read pid; do
            kill -SIGKILL "$pid" 2>/dev/null 2>/dev/null
        done
    fi
    
    sleep 1
    log "✅ 强制停止完成"
}

# 信号处理函数
cleanup() {
    echo "[$(date)] 收到停止信号，准备退出..."
    SHOULD_STOP=1
    
    # 立即终止Python子进程（如果存在）
    local download_pid=$(pgrep -f "python.*download_data.py" 2>/dev/null | head -n1)
    if [ ! -z "$download_pid" ]; then
        echo "[$(date)] 终止Python下载进程 (PID: $download_pid)..."
        kill -SIGTERM "$download_pid" 2>/dev/null
        sleep 2
        # 如果SIGTERM不生效，强制kill
        if kill -0 "$download_pid" 2>/dev/null; then
            echo "[$(date)] SIGTERM未生效，强制SIGKILL..."
            kill -SIGKILL "$download_pid" 2>/dev/null
        fi
    fi
    
    # 保险：再执行一次pkill确保清理干净
    sleep 1
    if pgrep -f "python.*download_data.py" >/dev/null 2>&1; then
        echo "[$(date)] 检测到残留进程，执行pkill清理..."
        pkill -9 -f "python.*download_data.py" 2>/dev/null
    fi
}

# 捕获停止信号
trap cleanup SIGTERM SIGINT

# 注意：PID文件由 schedule_download.sh 管理，不要覆盖
# echo $$ > "$SCRIPT_DIR/.download_script.pid"

# 激活虚拟环境
source ~/.bashrc 2>/dev/null || true
source "$PROJECT_DIR/.venv/bin/activate"
cd "$PROJECT_DIR"

# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# 日志函数
log() {
    echo "[$(date)] $1"
}

# 清理损坏的缓存
clean_corrupted_cache() {
    local local_dir="$1"
    local cache_dir="$local_dir/.cache/huggingface/download"
    
    log "🧹 开始清理损坏的缓存..."
    
    if [ ! -d "$cache_dir" ]; then
        log "  ⚠️  缓存目录不存在: $cache_dir"
        return
    fi
    
    # 清理所有 .metadata 文件（这些是断点续传的元数据）
    local metadata_count=$(find "$cache_dir" -name "*.metadata" 2>/dev/null | wc -l)
    if [ "$metadata_count" -gt 0 ]; then
        find "$cache_dir" -name "*.metadata" -delete 2>/dev/null
        log "  ✓ 已删除 $metadata_count 个 .metadata 文件"
    fi
    
    # 清理所有 .incomplete 文件（未完成的下载）
    local incomplete_count=$(find "$cache_dir" -name "*.incomplete" 2>/dev/null | wc -l)
    if [ "$incomplete_count" -gt 0 ]; then
        find "$cache_dir" -name "*.incomplete" -delete 2>/dev/null
        log "  ✓ 已删除 $incomplete_count 个 .incomplete 文件"
    fi
    
    # 清理所有 .lock 文件
    local lock_count=$(find "$cache_dir" -name "*.lock" 2>/dev/null | wc -l)
    if [ "$lock_count" -gt 0 ]; then
        find "$cache_dir" -name "*.lock" -delete 2>/dev/null
        log "  ✓ 已删除 $lock_count 个 .lock 文件"
    fi
    
    log "✅ 缓存清理完成，可以重新开始下载"
}

# 检查日志中是否有416错误，如果有则立即清理缓存
check_and_clean_416_error() {
    local local_dir="$1"
    local output_log=""

    # 优先检查 download_scheduled.log（同级目录）
    if [ -f "$SCRIPT_DIR/download_scheduled.log" ]; then
        output_log="$SCRIPT_DIR/download_scheduled.log"
    # 检查是否通过 nohup.out 输出（可能在启动目录）
    elif [ -f "$PROJECT_DIR/nohup.out" ]; then
        output_log="$PROJECT_DIR/nohup.out"
    # 检查 /tmp 下的日志（某些系统可能重定向到这里）
    elif [ -f "/tmp/download_data.log" ]; then
        output_log="/tmp/download_data.log"
    else
        # 没有找到日志文件，跳过416检查
        return 1
    fi

    # 检查最新的200行日志是否有416错误
    if tail -n 200 "$output_log" | grep -q "416 Requested Range Not Satisfiable"; then
        log "⚠️  检测到416错误（缓存损坏），立即清理缓存..."
        clean_corrupted_cache "$local_dir"
        return 0
    fi

    return 1
}

# 执行下载任务（单次下载，带重试逻辑）
run_download() {
    local retry_count=0
    local local_dir="/mnt/hdd_data/public_data/PhysicalAI-Autonomous-Vehicles-only-4-cam"

    # 初始化失败计数
    reset_failure_count > /dev/null

    while [ $retry_count -lt $MAX_RETRIES ]; do
        # 检查是否应该停止
        if [ "$SHOULD_STOP" -eq 1 ]; then
            log "收到停止信号，退出下载循环"
            return 0
        fi

        local current_failures=$(get_failure_count)
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log "尝试下载 (第 $((retry_count + 1)) 次，连续失败: $current_failures/$CONSECUTIVE_FAILURES_THRESHOLD)"
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # 前台运行下载进程（阻塞直到完成或被信号中断）
        python "$SCRIPT_DIR/download_data.py" --chunk_id -1 --local_dir "$local_dir"
        local exit_code=$?

        # 检查是否收到停止信号
        if [ "$SHOULD_STOP" -eq 1 ]; then
            log "收到停止信号，退出下载循环"
            return 0
        fi

        # 检查是否成功
        if [ $exit_code -eq 0 ]; then
            log "✅ 下载成功！"
            reset_failure_count > /dev/null  # 重置连续失败计数
            return 0
        else
            log "❌ 下载失败 (退出码: $exit_code)"
            local current_failures=$(increment_failure_count)

            # 检查是否有416错误（立即清理）
            if check_and_clean_416_error "$local_dir"; then
                log "🔧 已自动清理416错误相关的缓存文件"
                reset_failure_count > /dev/null  # 重置计数，因为已经清理了
            elif [ "$current_failures" -ge "$CONSECUTIVE_FAILURES_THRESHOLD" ]; then
                # 如果连续失败次数超过阈值，清理缓存
                log "⚠️  连续失败 $current_failures 次（达到阈值 $CONSECUTIVE_FAILURES_THRESHOLD），执行缓存清理..."
                clean_corrupted_cache "$local_dir"
                reset_failure_count > /dev/null  # 重置计数
            fi

            log "⏳ 等待60秒后重试..."

            # 可中断的等待
            for i in $(seq 1 60); do
                if [ "$SHOULD_STOP" -eq 1 ]; then
                    log "收到停止信号，退出下载循环"
                    return 0
                fi
                sleep 1
            done

            retry_count=$((retry_count + 1))
        fi
    done

    log "❌ 已达到最大重试次数 ($MAX_RETRIES)，下载失败！"
    return 1
}

# 检查当前时间是否允许下载（18:00-次日08:30）
# 返回0表示允许下载，返回1表示需要等待
is_download_window() {
    local current_hour=$(date +%-H)
    local current_min=$(date +%-M)
    local current_time=$((current_hour * 100 + current_min))

    # 允许下载时段：18:00-23:59 或 00:00-08:30
    if [ "$current_time" -ge 1800 ] || [ "$current_time" -lt 830 ]; then
        return 0
    else
        return 1
    fi
}

# 计算距离下一个下载窗口开始还有多少秒
seconds_until_next_download() {
    local current_hour=$(date +%-H)
    local current_min=$(date +%-M)
    local current_sec=$(date +%-S)
    local current_time=$((current_hour * 3600 + current_min * 60 + current_sec))
    
    # 下载窗口从18:00:00开始
    local target_time=$((18 * 3600))
    
    if [ "$current_time" -lt "$target_time" ]; then
        # 当前在08:30-18:00之间，等待到今天的18:00
        echo $((target_time - current_time))
    else
        # 当前已过18:00，等待到明天的18:00
        echo $((86400 - current_time + target_time))
    fi
}

# 主循环：处理定时任务
main_loop() {
    local download_pid=""
    
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "🚀 定时下载管理器已启动"
    log "📅 下载时段：每天 18:00 - 次日 08:30"
    log "⏸️  等待时段：08:30 - 18:00"
    log "🔧 连续失败 $CONSECUTIVE_FAILURES_THRESHOLD 次后自动清理缓存"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    while true; do
        # 检查是否应该停止
        if [ "$SHOULD_STOP" -eq 1 ]; then
            log "收到停止信号，退出..."
            break
        fi

        # 检查当前是否在下载时段
        if is_download_window; then
            # 在下载时段，检查是否有下载任务在运行
            if [ -z "$download_pid" ] || ! kill -0 "$download_pid" 2>/dev/null; then
                # 没有下载任务或任务已结束，启动新的下载
                if [ -n "$download_pid" ]; then
                    wait $download_pid 2>/dev/null
                    local exit_code=$?
                    
                    # 检查下载结果
                    if [ $exit_code -eq 0 ]; then
                        log "✅ 下载任务完成！"
                    else
                        log "⚠️  下载任务退出 (退出码: $exit_code)"
                    fi
                fi
                
                log "🚀 启动下载任务..."
                SHOULD_STOP=0
                run_download &
                download_pid=$!
                log "下载任务已启动，PID: $download_pid"
            fi
        else
            # 不在下载时段（08:30-18:00），需要等待
            if [ -n "$download_pid" ] && kill -0 "$download_pid" 2>/dev/null; then
                # 如果有下载任务在运行，停止它
                log "⏰ 到达等待时段（08:30），停止下载任务..."
                SHOULD_STOP=1
                
                # 第一步：尝试优雅停止（SIGTERM）
                log "  📍 步骤1/3: 发送SIGTERM信号..."
                if kill -0 "$download_pid" 2>/dev/null; then
                    kill -SIGTERM "$download_pid" 2>/dev/null
                    sleep 5
                fi
                
                # 第二步：如果进程还在，发送SIGKILL
                if [ -n "$download_pid" ] && kill -0 "$download_pid" 2>/dev/null; then
                    log "  📍 步骤2/3: 进程未响应，发送SIGKILL..."
                    kill -SIGKILL "$download_pid" 2>/dev/null
                    sleep 2
                fi
                
                # 第三步：保险措施，使用pkill强制清理所有相关进程
                if pgrep -f "python.*download_data.py" >/dev/null 2>&1; then
                    log "  📍 步骤3/3: 检测到残留进程，执行pkill强制清理..."
                    force_stop_download
                else
                    log "  ✅ 下载进程已停止"
                fi
                
                wait $download_pid 2>/dev/null
                SHOULD_STOP=0
                download_pid=""
                log "✅ 下载任务已停止，进入等待模式"
            fi
            
            # 计算距离18:00还有多久
            local wait_seconds=$(seconds_until_next_download)
            local wait_hours=$((wait_seconds / 3600))
            local wait_minutes=$(( (wait_seconds % 3600) / 60 ))
            log "⏸️  非下载时段（08:30-18:00），等待 ${wait_hours}小时${wait_minutes}分钟 后开始下载..."
            
            # 可中断的等待（每60秒检查一次是否收到停止信号）
            local waited=0
            while [ "$waited" -lt "$wait_seconds" ]; do
                if [ "$SHOULD_STOP" -eq 1 ]; then
                    log "收到停止信号，退出..."
                    break 2
                fi
                sleep 60
                waited=$((waited + 60))
            done
            continue
        fi

        # 每30秒检查一次时间窗口
        sleep 30
    done

    log "👋 定时下载管理器退出"
    rm -f "$SCRIPT_DIR/.download_scheduled.pid"
    exit 0
}

# 启动主循环
main_loop
