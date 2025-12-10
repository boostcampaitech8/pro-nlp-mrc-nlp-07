#!/bin/bash
# LLM Ensemble 실행 스크립트

# 기본 실행
echo "Running LLM Ensemble..."
python ensemble.py \
    --model Qwen/Qwen3-14B \

echo "Done! Check ensemble_qwen3-14b_output.csv for results."
