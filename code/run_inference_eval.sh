#!/bin/bash

# 로그 파일 이름 설정
LOG_FILE="./inference_eval_output.log"

# 실행할 Python 스크립트와 인자
# --hub_model_repo_id ljy-base \
# roberta-large-tapt-n8-hybrid
PYTHON_COMMAND="python inference.py \
    --output_dir ./outputs/inference_eval/ \
    --overwrite_output_dir \
    --dataset_name ../data/train_dataset/ \
    --top_k_retrieval 20 \
    --hub_model_repo_id roberta-large-tapt-n8-hybrid \
    --do_eval"

# --- 스크립트 실행 시작 ---

# 1. 파일에 시작 알림 기록 (새 파일 생성 및 덮어쓰기)
echo "--- Starting Inference at $(date) ---" > $LOG_FILE
echo "Command: $PYTHON_COMMAND" >> $LOG_FILE
echo "--- Running Output Below ---" >> $LOG_FILE

# 2. Python 명령어를 실행하고, 표준 출력(stdout)과 표준 에러(stderr)를 
#    모두 '$LOG_FILE'에 추가(&>>)합니다.
#    시작 부분은 이미 덮어썼으므로, 여기서는 추가 모드를 유지하여 로그 순서를 지킵니다.
$PYTHON_COMMAND &>> $LOG_FILE
# 또는: $PYTHON_COMMAND >> $LOG_FILE 2>&1

# 3. 파일에 종료 알림 기록 (추가)
echo "--- Inference Finished at $(date) ---" >> $LOG_FILE

echo "✅ Inference process completed. The output has overwritten '$LOG_FILE'."