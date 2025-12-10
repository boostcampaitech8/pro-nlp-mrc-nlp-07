#!/bin/bash
set -e

# === [경로 설정] ===
TAPT_DATA_PATH="../data/train_dataset"
DATA_PATH="../data/train_dataset_hybrid_neg_v1"  # 10만개 데이터 경로
TEST_DATA_PATH="../data/test_dataset"

# 초기 모델: KLUE RoBERTa Large (HuggingFace)
BASE_MODEL="klue/roberta-large"

# 저장 경로
TAPT_OUTPUT_DIR="./models/roberta_large_tapt_2025"
FINAL_QA_OUTPUT_DIR="./models/roberta_large_tapt_qa_2025"
SUBMISSION_DIR="./outputs/submission_tapt_roberta_2025"

echo "----------------------------------------------------------------"
echo " [Step 1] TAPT (Task-Adaptive Pre-Training)"
echo "   - Context 데이터를 이용해 MLM 학습을 진행합니다."
echo "----------------------------------------------------------------"

python TAPT.py \
  --output_dir ${TAPT_OUTPUT_DIR} \
  --model_name_or_path ${BASE_MODEL} \
  --dataset_name ${TAPT_DATA_PATH} \
  --do_train \
  --num_train_epochs 20 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --fp16 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --logging_steps 100 \
  --seed 2025

echo ""
echo "----------------------------------------------------------------"
echo " [Step 2] QA Fine-tuning"
echo "   - TAPT가 완료된 모델(${TAPT_OUTPUT_DIR})을 불러와 QA 학습을 합니다."
echo "----------------------------------------------------------------"

python train_roberta.py \
  --output_dir ${FINAL_QA_OUTPUT_DIR} \
  --model_name_or_path ${TAPT_OUTPUT_DIR} \
  --dataset_name ${DATA_PATH} \
  --do_train \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --fp16 \
  --logging_steps 500 \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --seed 2025

echo ""
echo "----------------------------------------------------------------"
echo " [Step 3] Inference"
echo "----------------------------------------------------------------"

python inference_roberta.py \
  --output_dir ${SUBMISSION_DIR} \
  --model_name_or_path ${FINAL_QA_OUTPUT_DIR} \
  --dataset_name ${TEST_DATA_PATH} \
  --do_predict \
  --max_seq_length 384 \
  --doc_stride 128 \
  --overwrite_output_dir

echo "✅ TAPT Pipeline Finished!"