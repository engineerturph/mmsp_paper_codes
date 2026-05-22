#!/bin/bash

# PIDs of the current 4 train_subset.py generation runs
PIDS="13793 13887 16194 16471"

echo "Waiting for processes ($PIDS) to finish..."
for pid in $PIDS; do
    while kill -0 $pid 2>/dev/null; do
        sleep 60
    done
done

echo "All generation runs have finished. Launching wandb sweep agents on all 4 GPUs..."

cd /home/yildirim26/kask_dusme_generation/windowed_generated_vs_nongenerated
SWEEP_ID="engineerturph-y-ld-z-technical-university/kask-generation/jziijvco"

for GPU in 0 1 2 3; do
    echo "Starting agent on GPU $GPU"
    CUDA_VISIBLE_DEVICES=$GPU nohup /home/yildirim26/kask_dusme_generation/.venv/bin/wandb agent $SWEEP_ID > sweep_agent_gpu${GPU}.log 2>&1 &
done

echo "Sweep agents launched in the background!"
