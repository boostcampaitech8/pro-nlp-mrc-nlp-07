#!/bin/bash

# 에 발생 시 중단
set -e

echo "===== [Step 1] Optimized Dataset (32k) Training Start (No Eval) ====="
# 수정: load_best_model_at_end 제거, evaluation_strategy 제거
python train.py \
  --output_dir ./models/roberta_reader_optimized \
  --dataset_name ../data/train_dataset_final_optimized \
  --do_train \
  --overwrite_output_dir \
  --model_name_or_path klue/roberta-large \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --logging_steps 500 \
  --save_strategy epoch \
  --save_total_limit 2

echo "===== [Step 2] Optimized Dataset (32k) Inference Start ====="
python inference.py \
  --output_dir ./outputs/submission_32000 \
  --dataset_name ../data/test_dataset/ \
  --model_name_or_path ./models/roberta_reader_optimized \
  --do_predict

echo "===== [Step 3] Full Dataset (84k) Training Start (No Eval) ====="
# 수정: load_best_model_at_end 제거, evaluation_strategy 제거
python train.py \
  --output_dir ./models/roberta_large_full_v2 \
  --dataset_name ../data/train_dataset_full_v2 \
  --do_train \
  --overwrite_output_dir \
  --model_name_or_path klue/roberta-large \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --logging_steps 500 \
  --save_strategy epoch \
  --save_total_limit 2

echo "===== [Step 4] Full Dataset (84k) Inference Start ====="
python inference.py \
  --output_dir ./outputs/submission_full_v2 \
  --dataset_name ../data/test_dataset/ \
  --model_name_or_path ./models/roberta_large_full_v2 \
  --do_predict

echo "===== All Jobs Finished! ====="
