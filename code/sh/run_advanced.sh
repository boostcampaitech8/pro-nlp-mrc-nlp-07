#!/bin/bash
set -e # 에러 발생 시 즉시 중단

# ====================================================
# [설정] 경로 및 파라미터 (여기를 본인 환경에 맞게 확인!)
# ====================================================

# 0. 초기 설정
INITIAL_MODEL="klue/roberta-large"              # 최초 베이스 모델
QA_TRAIN_DATA="../data/train_dataset_hybrid_neg_v1" # QA 학습 데이터
QA_TEST_DATA="../data/test_dataset"             # 최종 추론용 데이터

# 2. 모델이 저장될 경로들
TAPT_OUTPUT_DIR="./models/roberta_tapt_merged_3ep"
QA_OUTPUT_DIR="./models/roberta_advanced_final"
SUBMISSION_DIR="./outputs/submission_final_best"

# 3. 추론 설정
MAX_SEQ_LEN=384   
DOC_STRIDE=128

echo "========================================================"
echo "🚀 [Start] Final Boosting Process Started!"
echo "========================================================"



# ----------------------------------------------------
# [Step 2] TAPT (Train + Test) - 20 Epochs
# ----------------------------------------------------
echo "🚀 [Step 2] Starting TAPT (New Domain Adaptation)..."
python TAPT.py \
  --output_dir ${TAPT_OUTPUT_DIR} \
  --model_name_or_path ${INITIAL_MODEL} \
  --dataset_name ${QA_TRAIN_DATA} \
  --do_train \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --fp16 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --logging_steps 100 \
  --seed 2025 \
  --overwrite_output_dir

echo "✅ TAPT Finished!"

# ----------------------------------------------------
# [Step 3] QA Training (Advanced Strategy)
# ----------------------------------------------------
echo "🚀 [Step 3] Starting QA Training (Advanced)..."

# 주의: train_advanced.py 파일 내부에 설정을 강제로 박아넣은 코드를 사용하세요.
# (명령어 인자를 최소화하여 에러 방지)
python train_advanced.py \
  --output_dir ${QA_OUTPUT_DIR} \
  --model_name_or_path ${TAPT_OUTPUT_DIR} \
  --dataset_name ${QA_TRAIN_DATA} \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --logging_steps 500 \
  --fp16 \
  --seed 2024 \
  --overwrite_output_dir

echo "✅ QA Training Finished! (Best model automatically saved)"

# ----------------------------------------------------
# [Step 4] Inference
# ----------------------------------------------------
echo "🚀 [Step 4] Starting Inference..."

# QA_OUTPUT_DIR에는 이미 '가장 성능 좋았던 체크포인트'가 저장되어 있음
python inference_roberta.py \
  --output_dir ${SUBMISSION_DIR} \
  --model_name_or_path ${QA_OUTPUT_DIR} \
  --dataset_name ${QA_TEST_DATA} \
  --do_predict \
  --max_seq_length ${MAX_SEQ_LEN} \
  --doc_stride ${DOC_STRIDE} \
  --overwrite_output_dir

echo "🎉 All Done! Submission file ready at: ${SUBMISSION_DIR}"