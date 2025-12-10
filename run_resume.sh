#!/bin/bash

# [중요] set -e를 뺍니다. (경고 메시지 때문에 멈추는 것 방지)
# set -e 

# === [경로 설정] ===
DATA_PATH="../data/train_dataset_hybrid_neg_v1"  # 10만개 데이터
TEST_DATA_PATH="../data/test_dataset"

# [수정] 방금 학습이 완료된 TAPT 모델 경로
TAPT_OUTPUT_DIR="./models/roberta_large_tapt_qa"
FINAL_QA_OUTPUT_DIR="./models/roberta_large_tapt_qa_5EP"
SUBMISSION_DIR="./outputs/submission_tapt_roberta_5ep"

echo "================================================================"
echo " [Step 2] QA Fine-tuning (Start)"
echo "   - TAPT 모델(${TAPT_OUTPUT_DIR})을 불러와 QA 학습을 시작합니다."
echo "================================================================"

# [체크] TAPT 모델이 진짜 있는지 확인
if [ ! -d "$TAPT_OUTPUT_DIR" ]; then
  echo "❌ Error: TAPT 모델 폴더가 없습니다: ${TAPT_OUTPUT_DIR}"
  echo "Step 1이 제대로 저장되지 않았을 수 있습니다. 경로를 확인해주세요."
  exit 1
fi

# QA 학습 실행
python train_roberta.py \
  --output_dir ${FINAL_QA_OUTPUT_DIR} \
  --model_name_or_path ${TAPT_OUTPUT_DIR} \
  --dataset_name ${DATA_PATH} \
  --do_train \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-6 \
  --fp16 \
  --logging_steps 500 \
  --save_strategy "epoch" \
  --save_total_limit 2

echo ""
echo "================================================================"
echo " [Step 3] Inference (Start)"
echo "   - 학습된 QA 모델로 추론을 시작합니다."
echo "================================================================"

# 추론 실행
python inference_roberta.py \
  --output_dir ${SUBMISSION_DIR} \
  --model_name_or_path ${FINAL_QA_OUTPUT_DIR} \
  --dataset_name ${TEST_DATA_PATH} \
  --do_predict \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir

echo "✅ All Steps Finished Successfully!"