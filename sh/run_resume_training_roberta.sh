#!/bin/bash
set -e

# === [설정] ===
DATASET_PATH="../data/train_dataset_hybrid_neg_v1"

# 1. 불러올 체크포인트 경로 (여기서부터 학습 시작)
SOURCE_CHECKPOINT="CurtisJeon/klue-roberta-large-korquad_v1_qa"

# 2. 새로 학습된 모델이 저장될  (기존 폴더와 섞이지 않게 분리 추천)
NEW_OUTPUT_DIR="./models/CurtisJeon_3ep"
SUBMISSION_DIR="./outputs/submission_CurtisJeo_3ep"

echo ""
echo "========================================================================"
echo " [Step 1] Resume Training (3 Epochs) from Checkpoint"
echo "   - Source Model: ${SOURCE_CHECKPOINT}"
echo "   - Output Model: ${NEW_OUTPUT_DIR}"
echo "========================================================================"

# [핵심 수정]
# --model_name_or_path에 '체크포인트 '를 넣으면 그 가중치를 로드합니다.
python train_roberta.py \
  --output_dir ${NEW_OUTPUT_DIR} \
  --dataset_name ${DATASET_PATH} \
  --do_train \
  --overwrite_output_dir \
  --model_name_or_path ${SOURCE_CHECKPOINT} \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --logging_steps 500 \
  --save_strategy epoch \
  --save_total_limit 1
  --seed 2024

echo ""
echo "========================================================================"
echo " [Step 2] Inference with Newly Trained Model"
echo "========================================================================"

# 추론은 방금  학습을 마친 '새로운 모델 폴더'로 수행합니다.
python inference_roberta.py \
  --output_dir ${SUBMISSION_DIR} \
  --dataset_name ../data/test_dataset/ \
  --model_name_or_path ${NEW_OUTPUT_DIR} \
  --do_predict \
  --overwrite_output_dir

echo "✅ All processes finished!"
echo "Check output: ${SUBMISSION_DIR}/predictions.json"
