#!/bin/bash

# 에 발생 시 중단
set -e

# === [설정] ===
# 데이터 생성 스크립트 파일명
GEN_SCRIPT="make_concat_data.py"

# 생성될 데이터셋 경로
DATASET_DIR="../data/train_dataset_concat_context"


# 학습 모델 저장 경로
MODEL_OUTPUT_DIR="./models/uomnf97_concat_512"

# 최종 결과 저장 경로
SUBMISSION_DIR="./outputs/submission_concat_512"


echo "========================================================================"
echo " [Step 1] Creating Data Generation Script"
echo "========================================================================"
# 데이터 생성 실행
python make_concat_data.py


echo ""
echo "========================================================================"
echo " [Step 2] Training (Context Concatenation Strategy)"
echo "   - Model: uomnf97/klue-roberta-finetuned-korquad-v2"
echo "   - Max Seq Length: 512 (Increased)"
echo "========================================================================"

# 학습 실행 (Max Seq Length를 512로 늘리고, 배치를 줄임)
python train.py \
  --output_dir ${MODEL_OUTPUT_DIR} \
  --dataset_name ${DATASET_DIR} \
  --do_train \
  --overwrite_output_dir \
  --model_name_or_path uomnf97/klue-roberta-finetuned-korquad-v2 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --logging_steps 100 \
  --save_strategy epoch \
  --save_total_limit 1


echo ""
echo "========================================================================"
echo " [Step 3] Inference"
echo "   - Note: Inference also uses max_seq_length 512"
echo "========================================================================"

# 추론 실행 (여기서도 max_seq_length )
python inference.py \
  --output_dir ${SUBMISSION_DIR} \
  --dataset_name ../data/test_dataset/ \
  --model_name_or_path ${MODEL_OUTPUT_DIR} \
  --do_predict \
  --max_seq_length 512 \
  --overwrite_output_dir

echo "✅ All Jobs Finished Successfully!"
echo "Check output: ${SUBMISSION_DIR}/predictions.json"
