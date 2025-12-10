#!/bin/bash
set -e

# === [설정] ===
# 1. 밸런스 조절: 질문당 오답 4개 (정답 1.6만 : 오답 1.6만 = 1:1 황금비율)
DATASET_PATH="../data/train_dataset_final_balanced"
MODEL_OUTPUT_DIR="./models/uomnf97_korquad_balanced"
SUBMISSION_DIR="./outputs/submission_uomnf97_balanced"


echo ""
echo "========================================================================"
echo " [Step 2] Training uomnf97 model with Full Data"
echo "========================================================================"

# uomnf97 모델 + KorQuAD 증강 데이터 학습
python train.py \
  --output_dir ${MODEL_OUTPUT_DIR} \
  --dataset_name ${DATASET_PATH} \
  --do_train \
  --overwrite_output_dir \
  --model_name_or_path uomnf97/klue-roberta-finetuned-korquad-v2 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --logging_steps 500 \
  --save_strategy epoch \
  --save_total_limit 2


echo ""
echo "========================================================================"
echo " [Step 3] Inference"
echo "========================================================================"

python inference.py \
  --output_dir "./outputs/submission_uomnf97_ckpt21462" \
  --dataset_name ../data/test_dataset/ \
  --model_name_or_path "./models/uomnf97_korquad_balanced/checkpoint-21462" \
  --do_predict \
  --overwrite_output_dir

echo "✅ Finished!"
