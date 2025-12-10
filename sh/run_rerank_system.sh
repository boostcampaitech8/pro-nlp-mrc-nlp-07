#!/bin/bash

# 에러 발생 시 중단
set -e

echo ""
echo "========================================================"
echo " [Step 2] Reranker 파인튜닝 (Dongjin-kr Base)"
echo "========================================================"
# 생성된 데이터를 이용해 Dongjin-kr 모델을 미세 조정합니다. (Epoch 2, LR 5e-6)
python train_reranker.py

echo ""
echo "========================================================"
echo " [Step 3] 최종 추론 (New Reranker + Existing Reader)"
echo "========================================================"
# 학습된 리랭커와 기존 리더 모델(roberta_large_full_v2)을 합체하여 결과를 뽑습니다.
python inference.py \
  --output_dir ./outputs/submission_original_finetuned \
  --dataset_name ../data/test_dataset/ \
  --model_name_or_path ./models/roberta_large_full_v2 \
  --do_predict

echo ""
echo "✅ All processes finished successfully!"
echo "Check output: ./outputs/submission_original_finetuned/predictions.json"
