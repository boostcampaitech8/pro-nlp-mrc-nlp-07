#!/bin/bash
set -e  # 에러 발생 시 즉시 중단

# ====================================================
# [설정] 공통 파라미터
# ====================================================
DATASET_PATH="../data/test_dataset"
SEQ_LEN=384
STRIDE=128
TOP_K=20

echo "🚀 Starting Sequential Inference..."

# ----------------------------------------------------
# 1. 5 Epoch 모델 추론
# ----------------------------------------------------
echo ""
echo "========================================================"
echo " [Job 1] Running 5 Epoch Model..."
echo "========================================================"

python inference_roberta.py \
  --output_dir "./outputs/submission_tapt_roberta_5ep" \
  --model_name_or_path "./models/roberta_large_tapt_qa_5EP" \
  --dataset_name ${DATASET_PATH} \
  --do_predict \
  --max_seq_length ${SEQ_LEN} \
  --doc_stride ${STRIDE} \
  --top_k_retrieval ${TOP_K} \
  --overwrite_output_dir

echo " ✅ Job 1 Finished!"

# ----------------------------------------------------
# 2. 4 Epoch 체크포인트 추론
# ----------------------------------------------------
echo ""
echo "========================================================"
echo " [Job 2] Running 4 Epoch Checkpoint (8960)..."
echo "========================================================"

python inference_roberta.py \
  --output_dir "./outputs/submission_tapt_roberta_4ep" \
  --model_name_or_path "./models/roberta_large_tapt_qa_5EP/checkpoint-8960" \
  --dataset_name ${DATASET_PATH} \
  --do_predict \
  --max_seq_length ${SEQ_LEN} \
  --doc_stride ${STRIDE} \
  --top_k_retrieval ${TOP_K} \
  --overwrite_output_dir

echo " ✅ Job 2 Finished!"

echo ""
echo "🎉 All Inference Jobs Completed Successfully."