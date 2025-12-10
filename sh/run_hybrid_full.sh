#!/bin/bash

# 에 발생 시 스크립트 즉시 중단
set -e

# === [설정] ===
DATASET_DIR="../data/train_dataset_hybrid_neg_v2"
MODEL_OUTPUT_DIR="./models/roberta_hybrid_3ep_120000"
SUBMISSION_DIR="./outputs/submission_hybrid_3ep_120000"


echo "========================================================================"
echo " [Step 1] Hybrid Hard Negative Data Generation"
echo "   - Source: ES [Sparse] + DPR [Dense]"
echo "========================================================================"

# 데이터 생성 스크립트 실행
python make_hybrid_negative.py


echo ""
echo "========================================================================"
echo " [Step 2] Reader Training - 3 Epochs"
echo "   - Dataset: ${DATASET_DIR}"
echo "   - Model: klue/roberta-large"
echo "========================================================================"

# Reader 학습 (3 Epoch)
python train.py \
  --output_dir ${MODEL_OUTPUT_DIR} \
  --dataset_name ${DATASET_DIR} \
  --do_train \
  --overwrite_output_dir \
  --model_name_or_path klue/roberta-large \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --logging_steps 500 \
  --save_strategy epoch \
  --save_total_limit 2


echo ""
echo "========================================================================"
echo " [Step 3] Final Inference"
echo "   - Retriever: Hybrid [ES + DPR] -> Rerank [Dongjin]"
echo "========================================================================"

# 학습된 모델로 추론
python inference.py \
  --output_dir ${SUBMISSION_DIR} \
  --dataset_name ../data/test_dataset/ \
  --model_name_or_path ${MODEL_OUTPUT_DIR} \
  --do_predict \
  --overwrite_output_dir

echo ""
echo "✅ All Jobs Finished Successfully!"
echo "Check output: ${SUBMISSION_DIR}/predictions.json"
