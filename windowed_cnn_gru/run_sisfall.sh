#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 train.py \
  --data-path datasets/sisfall \
  --epochs 200 \
  --batch_size 500 \
  --window_size 256 \
  --stride 128 \
  --gpu "${GPU:-0}"
