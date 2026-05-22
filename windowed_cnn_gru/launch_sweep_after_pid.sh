#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:?Usage: ./launch_sweep_after_pid.sh <pid_to_wait_for> <gpu> <sweep_id> [python_path]}"
GPU_ID="${2:?Usage: ./launch_sweep_after_pid.sh <pid_to_wait_for> <gpu> <sweep_id> [python_path]}"
SWEEP_ID="${3:?Usage: ./launch_sweep_after_pid.sh <pid_to_wait_for> <gpu> <sweep_id> [python_path]}"
PYTHON_BIN="${4:-$HOME/kask_dusme_generation/.venv/bin/python}"

cd "$(dirname "$0")"
mkdir -p logs

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

CUDA_VISIBLE_DEVICES="$GPU_ID" nohup "$PYTHON_BIN" -m wandb agent "$SWEEP_ID" \
  > "logs/sweep_agent_gpu${GPU_ID}.log" 2>&1 &
echo "gpu=${GPU_ID} pid=$!"
