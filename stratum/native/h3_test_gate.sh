#!/usr/bin/env zsh
# h3_test_gate.sh — 边界3合规的大模型测试门禁 + 清理
#
# 用法:
#   h3_test_gate.sh check              检查系统是否允许启动大模型测试
#   h3_test_gate.sh clean <model...>   测试后立即释放指定模型文件的页缓存
#
# 协议（对齐 AGENTS.md 边界3）:
#   1. free+inactive >= H3_MIN_FREE_GB (默认8GB) 才允许启动
#   2. swap used 占比 >= H3_MAX_SWAP_PCT (默认60%) 时拒绝启动
#   3. 大测试结束后必须立刻对用过的每个模型文件执行 clean
set -euo pipefail

MIN_FREE_GB="${H3_MIN_FREE_GB:-8}"
MAX_SWAP_PCT="${H3_MAX_SWAP_PCT:-60}"

mem_stats() {
    local stats=$(vm_stat | python3 -c "
import sys
free=inact=0
for line in sys.stdin:
    if line.startswith('Pages free:'): free=int(line.split(':')[1].strip().rstrip('.'))
    if line.startswith('Pages inactive:'): inact=int(line.split(':')[1].strip().rstrip('.'))
print(f'{free*16384/1e9:.2f} {(free+inact)*16384/1e9:.2f}')
")
    FREE_GB=$(echo $stats | cut -d' ' -f1)
    AVAIL_GB=$(echo $stats | cut -d' ' -f2)

    SWAP_USED=$(sysctl -n vm.swapusage | sed 's/.*used = \([0-9.]*\)M.*/\1/')
    SWAP_TOTAL=$(sysctl -n vm.swapusage | sed 's/.*total = \([0-9.]*\)M.*/\1/')
    SWAP_PCT=$(python3 -c "print(int($SWAP_USED / $SWAP_TOTAL * 100))")
}

case "${1:-}" in
check)
    mem_stats
    echo "free: ${FREE_GB}GB | avail(free+inactive): ${AVAIL_GB}GB | swap: ${SWAP_USED}M/${SWAP_TOTAL}M (${SWAP_PCT}%)"
    if (( $(echo "$AVAIL_GB < $MIN_FREE_GB" | bc -l) )); then
        echo "❌ 拒绝启动: 可用内存 ${AVAIL_GB}GB < 门槛 ${MIN_FREE_GB}GB (AGENTS.md 边界3)"
        exit 1
    fi
    if (( SWAP_PCT >= MAX_SWAP_PCT )); then
        echo "❌ 拒绝启动: swap 占用 ${SWAP_PCT}% >= 上限 ${MAX_SWAP_PCT}%"
        exit 1
    fi
    echo "✅ 允许启动"
    ;;
clean)
    shift
    for f in "$@"; do
        python3 "$(dirname "$0")/../tools/drop_page_cache.py" "$f" | tail -1
    done
    ;;
*)
    echo "usage: $0 check | $0 clean <model...>"
    exit 1
    ;;
esac
