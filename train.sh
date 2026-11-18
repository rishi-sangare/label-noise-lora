#!/usr/bin/env bash
# Train one LoRA adapter. Every run uses identical settings; only data and seed change.
# Usage: ./train.sh <variant-dir-under-data> <seed>
set -euo pipefail
VARIANT=$1
SEED=$2
NAME="${VARIANT}-s${SEED}"
mkdir -p logs
.venv/bin/python -m mlx_lm lora \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --train \
  --data "data/${VARIANT}" \
  --fine-tune-type lora \
  --num-layers 16 \
  --batch-size 8 \
  --iters 500 \
  --learning-rate 1e-4 \
  --mask-prompt \
  --steps-per-report 50 \
  --steps-per-eval 100 \
  --val-batches 25 \
  --max-seq-length 512 \
  --seed "${SEED}" \
  --adapter-path "adapters/${NAME}" 2>&1 | tee "logs/${NAME}.log"
.venv/bin/python evaluate.py --name "${NAME}" --adapter "adapters/${NAME}"
