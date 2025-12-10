#!/bin/bash
set -e

# === [경로 설정] ===
TEST_DATA_PATH="./data/test_dataset"
SUBMISSION_DIR="./outputs/submission_tapt_roberta_2025"

# 모델 설정
READER_MODEL_NAME="NLP-07-ODQA/roberta-large-tapt-n8-hybrid"
DENSE_MODEL_NAME="BAAI/bge-m3"
RERANK_MODEL_NAME="BAAI/bge-reranker-v2-m3"


echo ""
echo "----------------------------------------------------------------"
echo " [Step 3] Inference"
echo "----------------------------------------------------------------"

python inference_roberta.py \
  --output_dir ${SUBMISSION_DIR} \
  --dataset_name ${TEST_DATA_PATH} \
  --model_name_or_path ${READER_MODEL_NAME} \
  --dense_model_name ${DENSE_MODEL_NAME} \
  --rerank_model_path ${RERANK_MODEL_NAME} \
  --do_predict

echo "✅ TAPT Pipeline Finished!"