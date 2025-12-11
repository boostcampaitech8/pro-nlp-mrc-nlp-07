#!/bin/bash

# Qwen3 Hybrid Retrieval Inference 실행 스크립트
# 1. Validation 데이터셋으로 평가 (do_eval)
# 2. Test 데이터셋으로 예측 (do_predict)

set -e  # 에러 발생 시 스크립트 중단

# 스크립트가 있는 디렉토리로 이동 (code 디렉토리)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 가상환경 활성화 (필요한 경우 주석 해제하고 경로 수정)
# source /path/to/venv/bin/activate
# 또는 conda 환경 사용 시:
# conda activate your_env_name

echo "=========================================="
echo "Starting Inference Pipeline"
echo "=========================================="
echo ""

# 1. Validation 데이터셋으로 평가
echo "=========================================="
echo "Step 1: Evaluating on validation dataset"
echo "=========================================="
python inference.py \
  --output_dir ./outputs/eval_qwen3_hybrid_chunk_top_10 \
  --dataset_name ../data/train_dataset \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --do_eval \
  --use_hybrid_retrieval \
  --top_k_retrieval 10

if [ $? -ne 0 ]; then
    echo "Error: Step 1 failed!"
    exit 1
fi

echo ""
echo "Step 1 completed successfully!"
echo ""

# 2. Test 데이터셋으로 예측
echo "=========================================="
echo "Step 2: Predicting on test dataset"
echo "=========================================="
python inference.py \
  --output_dir ./outputs/test_qwen3_hybrid_chunk_top_10 \
  --dataset_name ../data/test_dataset \
  --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
  --do_predict \
  --use_hybrid_retrieval \
  --top_k_retrieval 10

if [ $? -ne 0 ]; then
    echo "Error: Step 2 failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "All steps completed successfully!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Evaluation: ./outputs/eval_qwen3_hybrid_chunk_top_10"
echo "  - Predictions: ./outputs/test_qwen3_hybrid_chunk_top_10"
