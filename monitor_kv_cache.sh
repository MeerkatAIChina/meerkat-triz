#!/bin/bash
# Meerkat KV cache 监控脚本
# 检查 vLLM KV cache 使用率、请求队列、内存，超阈值告警
# 用法:
#   monitor_kv_cache.sh --once            # 单次检查
#   monitor_kv_cache.sh --loop [秒数]      # 持续监控(默认 300s)

METRICS_URL="http://127.0.0.1:8888/metrics"
LOG="/home/chinux/jupyterlab/meerkatai/kv_cache_monitor.log"
INTERVAL="${2:-300}"

# ===== 阈值(可调) =====
KV_WARN=85        # KV cache 使用率警告线(%)
KV_CRIT=95        # KV cache 使用率危险线(%)
MEM_WARN_GB=10    # 可用内存警告线(GB)
WAIT_WARN=5       # 等待队列警告线(个)

log() {
    local msg="[$(date '+%F %T')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG"
}

# 从 metrics 提取单个指标值
get_metric() {
    echo "$1" | grep -oE "vllm:${2}\\{[^}]*\\} [0-9.]+" | grep -oE '[0-9.]+$' | head -1
}

check_once() {
    local metrics kv_usage running waiting avail_gb
    metrics=$(curl -s -m 5 "$METRICS_URL" 2>/dev/null)
    if [ -z "$metrics" ]; then
        log "!! CRIT 无法获取 metrics（vLLM 可能未运行）"
        return 1
    fi

    kv_usage=$(get_metric "$metrics" "kv_cache_usage_perc")
    # 指标值是小数(0.44 = 44%)，转百分比
    [ -n "$kv_usage" ] && kv_usage=$(awk "BEGIN{printf \"%.1f\", $kv_usage * 100}")
    running=$(get_metric "$metrics" "num_requests_running")
    waiting=$(get_metric "$metrics" "num_requests_waiting")
    avail_gb=$(free -g | awk '/Mem:/{print $7}')

    local level="OK"
    local warn_msg=""

    # KV cache 使用率判断
    if [ -n "$kv_usage" ]; then
        if awk "BEGIN{exit !($kv_usage >= $KV_CRIT)}"; then
            level="CRIT"; warn_msg="KV cache 使用率 ${kv_usage}% >= ${KV_CRIT}% 危险线"
        elif awk "BEGIN{exit !($kv_usage >= $KV_WARN)}"; then
            level="WARN"; warn_msg="KV cache 使用率 ${kv_usage}% >= ${KV_WARN}% 警告线"
        fi
    fi

    # 等待队列判断
    if [ -n "$waiting" ] && awk "BEGIN{exit !($waiting >= $WAIT_WARN)}"; then
        [ "$level" = "OK" ] && level="WARN"
        warn_msg="${warn_msg:+${warn_msg}; }等待队列 ${waiting} 个请求排队"
    fi

    # 可用内存判断
    if [ -n "$avail_gb" ] && [ "$avail_gb" -lt "$MEM_WARN_GB" ]; then
        [ "$level" = "OK" ] && level="WARN"
        warn_msg="${warn_msg:+${warn_msg}; }可用内存仅 ${avail_gb}GB < ${MEM_WARN_GB}GB"
    fi

    log "[$level] KV:${kv_usage:-?}% | run:${running:-?} | wait:${waiting:-?} | mem:${avail_gb:-?}GB${warn_msg:+ | $warn_msg}"
}

case "$1" in
    --once)
        check_once
        ;;
    --loop)
        log "启动 KV cache 持续监控（间隔 ${INTERVAL}s，日志 $LOG）"
        while true; do
            check_once
            sleep "$INTERVAL"
        done
        ;;
    *)
        echo "用法: $0 [--once | --loop 秒数]"
        exit 1
        ;;
esac
