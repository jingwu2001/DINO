#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/valve_coco [pretrained_checkpoint.pth]"
  exit 1
fi

DATA_ROOT=$1
PRETRAIN=${2:-}
OUTPUT_DIR=logs/valve/R50-MS4

mkdir -p "${OUTPUT_DIR}"

CMD=(python main.py -c config/DINO/DINO_valve_4scale.py --coco_path "${DATA_ROOT}" --output_dir "${OUTPUT_DIR}")

if [ -n "${PRETRAIN}" ]; then
  CMD+=(--pretrain_model_path "${PRETRAIN}" --finetune_ignore label_enc.weight class_embed)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
