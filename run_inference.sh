#!/bin/bash
set -e

# === [경로 설정] ===
TEST_DATA_PATH="./data/test_dataset"
MODEL_NAME="NLP-07-ODQA/roberta-large-tapt-n8-hybrid"
SUBMISSION_DIR="./outputs/submission_tapt_roberta_2025"


echo ""
echo "----------------------------------------------------------------"
echo " [Step 3] Inference"
echo "----------------------------------------------------------------"

python inference_roberta.py \
  --output_dir ${SUBMISSION_DIR} \
  --model_name_or_path ${MODEL_NAME} \
  --dataset_name ${TEST_DATA_PATH} \
  --do_predict

echo "✅ TAPT Pipeline Finished!"