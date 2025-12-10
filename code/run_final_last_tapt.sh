#!/bin/bash

# [중요] set -e를 뺍니다. (경고 메시지 때문에 멈추는 것 방지)
# set -e 

# === [경로 설정] ===
DATA_PATH="../data/train_dataset_hybrid_neg_v1"  # 10만개 데이터
TEST_DATA_PATH="../data/test_dataset"
INITIAL_MODEL="klue/roberta-large"  

TAPT_OUTPUT_DIR="./models/roberta_tapt_last_10ep"
FINAL_QA_OUTPUT_DIR="./models/roberta_last_tapt_qa_3EP"
SUBMISSION_DIR="./outputs/submission_last_tapt_roberta_3ep"
QA_TRAIN_DATA="../data/train_dataset"
echo "================================================================"
echo " [Step 2] QA Fine-tuning (Start)"
echo "   - TAPT 모델(${TAPT_OUTPUT_DIR})을 불러와 QA 학습을 시작합니다."
echo "================================================================"

python TAPT.py \
  --output_dir ${TAPT_OUTPUT_DIR} \
  --model_name_or_path ${INITIAL_MODEL} \
  --dataset_name ${QA_TRAIN_DATA} \
  --do_train \
  --num_train_epochs 10 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --fp16 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --logging_steps 500 \
  --overwrite_output_dir



# QA 학습 실행
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
  --save_total_limit 1

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
  --top_k_retrieval 20 \
  --overwrite_output_dir

echo "✅ All Steps Finished Successfully!"