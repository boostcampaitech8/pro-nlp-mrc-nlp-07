#!/bin/bash

# 에러 발생 시 중단
set -e

# === [설정] ===
# 1. DPR 학습 스크립        (아까 만든 파일명)
DPR_TRAIN_SCRIPT="train_dpr_lora.py"

# 2. DPR 모델이 저장될 경로 (train_dpr_lora_final.py 내부 설정과 같아야 함)
DPR_MODEL_PATH="./models/dpr_lora_finetuned"

# 3. 추론에 사용할 Reader 모델 체크포인트 (사용자 지정)
READER_CHECKPOINT="./models/uomnf97_korquad_balanced/checkpoint-21462"

# 4. 결과 저장 경로
SUBMISSION_DIR="./outputs/submission_dpr_finetuned_ckpt21462"


echo "========================================================================"
echo " [Step 1] DPR (Dense Retriever) Fine-tuning with LoRA"
echo "========================================================================"

# 1. DPR 학습 실행
if [ -f "$DPR_TRAIN_SCRIPT" ]; then
    python $DPR_TRAIN_SCRIPT
else
    echo "Error: $DPR_TRAIN_SCRIPT 파일이 없습니."
    exit 1
fi


echo ""
echo "========================================================================"
echo " [Step 3] Final Inference using Reader Checkpoint"
echo "   - Reader: ${READER_CHECKPOINT}"
echo "   - Retriever: Hybrid (ES + Fine-tuned DPR)"
echo "========================================================================"

# 3. 추론 실행
# inference.py 수정된 retrieval.py를 불러오므로, 자동으로 새 DPR을 사용합니다.
python inference.py \
  --output_dir ${SUBMISSION_DIR} \
  --dataset_name ../data/test_dataset/ \
  --model_name_or_path ${READER_CHECKPOINT} \
  --do_predict \
  --overwrite_output_dir

echo ""
echo "✅ All Jobs Finished!"
echo "Check output: ${SUBMISSION_DIR}/predictions.json"
