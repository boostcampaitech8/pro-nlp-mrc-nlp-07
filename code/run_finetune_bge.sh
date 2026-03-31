#!/bin/bash

# --- 스크립트 설정 ---

# 1. 로그 파일 이름 설정 (시간을 포함하여 실행할 때마다 새로운 파일 생성)
LOG_FILE="./logs/finetune_bge_$(date +%Y%m%d_%H%M%S).log"

# 2. 모델 학습 스크립트 경로 설정
PYTHON_SCRIPT="finetune_bge.py"

# 로그 디렉토리가 없으면 생성
mkdir -p ./logs

echo "--- Dense Retriever Fine-tuning Start ---" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

# Python 스크립트 실행
# 1. 'tee -a'를 사용하여 출력을 화면(STDOUT)과 로그 파일에 동시 기록합니다.
# 2. '2>&1'를 사용하여 오류 출력(STDERR)까지 모두 표준 출력(STDOUT)으로 리다이렉션합니다.
python "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"

# 종료 메시지
echo "----------------------------------------" | tee -a "$LOG_FILE"
echo "Fine-tuning process finished." | tee -a "$LOG_FILE"