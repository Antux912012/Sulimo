#!/usr/bin/env bash
# Súlimo i915 RC6 residency and CPU core pinning verification script ($BATT, A13, §7 M4).
set -euo pipefail

DURATION=30
PID=""
TEST_MODE=0

usage() {
    echo "Usage: $0 [--duration <sec>] [--pid <pid>] [--test]"
    echo "  --duration <sec>   Duration in seconds to monitor RC6 residency (default: 30)"
    echo "  --pid <pid>        Process ID of Sulimo engine to verify thread core pinning"
    echo "  --test             Quick test mode (samples for 3 seconds)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --pid)
            PID="$2"
            shift 2
            ;;
        --test)
            TEST_MODE=1
            DURATION=3
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Locate i915 rc6_residency_ms sysfs entry
RC6_PATH=""
for candidate in \
    /sys/class/drm/card*/gt/gt0/rc6_residency_ms \
    /sys/class/drm/card*/power/rc6_residency_ms \
    /sys/kernel/debug/dri/*/i915_rc6_residency; do
    if [[ -f "$candidate" && -r "$candidate" ]]; then
        RC6_PATH="$candidate"
        break
    fi
done

if [[ -z "$RC6_PATH" ]]; then
    echo "[-] Error: No readable rc6_residency_ms sysfs file found under /sys/class/drm/ or /sys/kernel/debug/dri/"
    exit 1
fi

echo "==> Súlimo iGPU RC6 Residency & Core Pinning Monitor"
echo "    RC6 sysfs node: $RC6_PATH"
echo "    Sample duration: ${DURATION}s"

# If PID is provided, verify decode thread pinning to physical cores (A10, §7 M4)
if [[ -n "$PID" ]]; then
    if [[ ! -d "/proc/$PID" ]]; then
        echo "[-] Error: Process ID $PID does not exist."
        exit 1
    fi

    echo "==> Verifying thread core affinity for PID $PID:"
    echo "    Task ID   | Thread Name          | Cpus_allowed_list"
    echo "    --------- | -------------------- | -----------------"
    for task_status in /proc/"$PID"/task/*/status; do
        if [[ -f "$task_status" ]]; then
            tid=$(basename "$(dirname "$task_status")")
            tname=$(grep '^Name:' "$task_status" | awk '{print $2}')
            allowed=$(grep '^Cpus_allowed_list:' "$task_status" | awk '{print $2}')
            printf "    %-9s | %-20s | %s\n" "$tid" "$tname" "$allowed"
        fi
    done
fi

# Initial sample
RC6_START=$(cat "$RC6_PATH")
TIME_START=$(date +%s%3N)

echo "==> Sampling RC6 residency for ${DURATION}s..."
sleep "$DURATION"

# Final sample
RC6_END=$(cat "$RC6_PATH")
TIME_END=$(date +%s%3N)

RC6_DELTA=$(( RC6_END - RC6_START ))
TIME_DELTA=$(( TIME_END - TIME_START ))

if [[ $TIME_DELTA -le 0 ]]; then
    echo "[-] Error: Invalid time delta."
    exit 1
fi

# Calculate residency percentage
RESIDENCY_PCT=$(( (RC6_DELTA * 100) / TIME_DELTA ))

echo "--------------------------------------------------------"
echo "Results:"
echo "    Wall time elapsed: ${TIME_DELTA} ms"
echo "    iGPU RC6 deep sleep: ${RC6_DELTA} ms"
echo "    RC6 residency: ${RESIDENCY_PCT}%"
echo "--------------------------------------------------------"

if [[ $RESIDENCY_PCT -ge 90 ]]; then
    echo "[+] PASS: iGPU RC6 residency is ${RESIDENCY_PCT}% (>= 90% target, A13 fulfilled)."
    exit 0
else
    echo "[-] FAIL: iGPU RC6 residency is ${RESIDENCY_PCT}% (< 90% target)."
    exit 1
fi
