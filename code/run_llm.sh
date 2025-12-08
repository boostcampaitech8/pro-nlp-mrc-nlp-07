#!/bin/bash
set -e  # 에러 발생 시 즉시 중단

# ====================================================
# [설정] 경로 및 로그
# ====================================================
LOG_FILE="./llm_pipeline.log"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
echo "================================================================"
echo " 🚀 LLM Pipeline Start (Training + Inference)"
echo "    - Log file: ${LOG_FILE}"
echo "================================================================" | tee -a ${LOG_FILE}

# ----------------------------------------------------
# Step 1. LLM Fine-tuning (Train)
# ----------------------------------------------------
echo "" | tee -a ${LOG_FILE}
echo " [Step 1] Starting Training (train_llm.py)..." | tee -a ${LOG_FILE}
date | tee -a ${LOG_FILE}

# 학습 실행
python train_llm.py 2>&1 | tee -a ${LOG_FILE}

echo " ✅ Training Finished!" | tee -a ${LOG_FILE}

# ----------------------------------------------------
# Step 2. LLM Inference (Predict)
# ----------------------------------------------------
echo "" | tee -a ${LOG_FILE}
echo " [Step 2] Starting Inference (inference_llm.py)..." | tee -a ${LOG_FILE}
date | tee -a ${LOG_FILE}

# 추론 실행
python inference_llm.py 2>&1 | tee -a ${LOG_FILE}

echo " ✅ Inference Finished!" | tee -a ${LOG_FILE}

echo "" | tee -a ${LOG_FILE}
echo "================================================================" | tee -a ${LOG_FILE}
echo " 🎉 All Jobs Completed Successfully." | tee -a ${LOG_FILE}
echo "    - Check 'submission_llm.csv' for results."
echo "================================================================" | tee -a ${LOG_FILE}