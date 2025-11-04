#!/bin/bash
set -e

export TOKENIZERS_PARALLELISM=false
# Go to the repo root
cd "$(dirname "$0")/"
export PYTHONPATH="$(pwd)/:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1

# Paths
CFG="config/GroundingDINO_SwinB_cfg.py"
DATA_JSON="dataset_meta_visdrone.json"
PRETRAIN="../vlmtest/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
OUT_DIR="../vlmtest/GroundingDINO/output/visdrone_swinb_aggresive_finetune"

# Launch distributed training
python -m torch.distributed.launch --nproc_per_node=2 main.py \
  -c "$CFG" \
  --datasets "$DATA_JSON" \
  --output_dir "$OUT_DIR" \
  --pretrain_model_path "$PRETRAIN" \
  --num_workers 32 \
  --options \
    batch_size=8 \
    use_coco_eval=False
