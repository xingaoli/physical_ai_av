#!/bin/bash

# 合并的定时下载脚本
# 功能: 
#   - 立即启动下载任务
#   - 每天18:00自动重启下载任务
#   - 每天08:30自动停止下载任务
#   - 下载失败自动重试（最多9999次）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
LOG_FILE="$SCRIPT_DIR/output.log"
MAX_RETRIES=9999
SHOULD_STOP=0
DOWNLOAD_PID=""

# 信号处理函数
cleanup() {
    echo "[$(date)] 收到停止信号，准备退出..." | tee -a "$LOG_FILE"
    SHOULD_STOP=1
    if [ ! -z "$DOWNLOAD_PID" ] && kill -0 "$DOWNLOAD_PID" 2>/dev/null; then
        echo "[$(date)] 终止下载子进程 (PID: $DOWNLOAD_PID)..." | tee -a "$LOG_FILE"
        kill -TERM "$DOWNLOAD_PID" 2>/dev/null
    fi
}

# 捕获停止信号
trap cleanup SIGTERM SIGINT

# 记录主脚本PID
echo $$ > "$SCRIPT_DIR/.download_script.pid"

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
    echo "[$(date)] $1" | tee -a "$LOG_FILE"
}

# 执行下载任务
run_download() {
    local retry_count=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        # 检查是否应该停止
        if [ "$SHOULD_STOP" -eq 1 ]; then
            log "收到停止信号，退出下载循环"
            return 0
        fi
        
        log "尝试下载 (第 $((retry_count + 1)) 次)..."
        
        # 后台运行下载进程
        python tools/download_data.py --chunk_id -1 --local_dir /mnt/hdd_data/public_data/PhysicalAI-Autonomous-Vehicles-only-4-cam &
        DOWNLOAD_PID=$!
        
        # 等待子进程完成
        wait $DOWNLOAD_PID
        local exit_code=$?
        DOWNLOAD_PID=""
        
        # 检查是否收到停止信号
        if [ "$SHOULD_STOP" -eq 1 ]; then
            log "收到停止信号，退出下载循环"
            return 0
        fi
        
        # 检查是否成功
        if [ $exit_code -eq 0 ]; then
            log "✅ 下载成功！"
            return 0
        else
            log "❌ 下载失败 (退出码: $exit_code)，60秒后重试..."
            
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

# 主循环：处理定时任务
main_loop() {
    log "定时下载管理器已启动"
    log "下载任务将在每天 18:00 重启，08:30 停止"
    
    # 立即启动首次下载
    log "启动首次下载任务..."
    run_download &
    local download_main_pid=$!
    
    while true; do
        CURRENT_TIME=$(date +%H%M)
        
        # 18:00 重启下载任务
        if [ "$CURRENT_TIME" = "1800" ]; then
            log "到达 18:00，重启下载任务..."
            if kill -0 "$download_main_pid" 2>/dev/null; then
                kill -TERM "$download_main_pid" 2>/dev/null
                sleep 5
                if kill -0 "$download_main_pid" 2>/dev/null; then
                    kill -KILL "$download_main_pid" 2>/dev/null
                fi
            fi
            wait $download_main_pid 2>/dev/null
            
            run_download &
            download_main_pid=$!
            
            sleep 60  # 避免重复触发
        fi
        
        # 08:30 停止下载任务
        if [ "$CURRENT_TIME" = "0830" ]; then
            log "到达 08:30，停止下载任务..."
            if kill -0 "$download_main_pid" 2>/dev/null; then
                kill -TERM "$download_main_pid" 2>/dev/null
                sleep 5
                if kill -0 "$download_main_pid" 2>/dev/null; then
                    kill -KILL "$download_main_pid" 2>/dev/null
                fi
            fi
            wait $download_main_pid 2>/dev/null
            SHOULD_STOP=1
            log "下载任务已停止"
            
            # 清理PID文件
            rm -f "$SCRIPT_DIR/.download_script.pid"
            
            # 退出主脚本
            log "定时下载管理器退出"
            exit 0
        fi
        
        # 检查下载进程是否意外退出，如果是则重启
        if ! kill -0 "$download_main_pid" 2>/dev/null; then
            wait $download_main_pid 2>/dev/null
            local exit_code=$?
            if [ $exit_code -ne 0 ]; then
                log "下载进程意外退出 (退出码: $exit_code)，准备重启..."
                sleep 10
                SHOULD_STOP=0
                run_download &
                download_main_pid=$!
            fi
        fi
        
        sleep 10
    done
}

# 启动主循环
main_loop
