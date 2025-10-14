#!/bin/bash
set -e

# Go to the repo root
cd "$(dirname "$0")/"
export PYTHONPATH="$(pwd)/:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1

# Paths
CFG="config/GroundingDINO_SwinB_cfg.py"
DATA_JSON="dataset_meta_kitti.json"
PRETRAIN="../vlmtest/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
OUT_DIR="../vlmtest/GroundingDINO/outputs/kitti_swinb_finetune"

# Launch distributed training
python -m torch.distributed.launch --nproc_per_node=2 main.py \
  -c "$CFG" \
  --datasets "$DATA_JSON" \
  --output_dir "$OUT_DIR" \
  --pretrain_model_path "$PRETRAIN" \
  --num_workers 8 \
  --options \
    text_encoder_type=bert-base-uncased \
    backbone='swin_B_384_22k' \
    lr=1e-4 \
    batch_size=4 \
    epochs=1 \
    use_coco_eval=False
