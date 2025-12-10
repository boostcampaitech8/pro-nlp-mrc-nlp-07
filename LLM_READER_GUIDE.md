# LLM Reader Training Guide

이 실험은 BERT QuestionAnswering 방식에서 CasualLM (Qwen3) 방식으로 전환된 Reader 모델입니다.

## 주요 변경 사항

### 1. 모델 아키텍처
- **이전**: `AutoModelForQuestionAnswering` (BERT 기반)
- **현재**: `AutoModelForCausalLM` (Qwen3 기반)

### 2. 데이터 형식
- **이전**: Question + Context → Start/End Position
- **현재**: Instruction Prompt → Answer Generation

### 3. 프롬프트 템플릿 (Instruction 형식)
```
### 지시사항
주어진 문맥을 읽고 질문에 답하세요.

### 문맥
{context}

### 질문
{question}

### 답변
{answer}
```

### 4. Loss 계산
- **답변 부분만 loss 계산** (프롬프트 부분은 -100으로 마스킹)
- EM 평가를 위해 답변 외 다른 텍스트가 생성되지 않도록 설정

## 파일 구조

```
llm-reader/
├── train.py                 # CasualLM 학습 스크립트 (수정됨)
├── arguments.py             # 인자 정의 (LLM 관련 인자 추가)
├── data_formatter.py        # 프롬프트 생성 모듈 (신규)
├── data_collator.py         # Answer-only loss collator (신규)
├── test_setup.py            # 설정 테스트 스크립트 (신규)
└── inference.py             # 추론 스크립트 (아직 미수정)
```

## 사용 방법

### 1. 설정 테스트
```bash
python test_setup.py
```

### 2. 학습 실행
```bash
python train.py \
  --model_name_or_path Qwen/Qwen3-8B \
  --dataset_name ./data/train_dataset \
  --output_dir ./models/llm-reader \
  --do_train \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --logging_steps 100 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --load_best_model_at_end False \
  --prompt_template_type instruction \
  --max_seq_length 512 \
  --max_answer_tokens 50 \
  --fp16 \
  --gradient_checkpointing True \
  --optim adamw_torch \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --report_to none
```

### 3. 평가 실행
```bash
python train.py \
  --model_name_or_path ./models/llm-reader \
  --dataset_name ./data/train_dataset \
  --output_dir ./models/llm-reader \
  --do_eval \
  --per_device_eval_batch_size 8 \
  --prompt_template_type instruction \
  --max_seq_length 512

python inference.py \
  --model_name_or_path ./models/llm-reader \
  --dataset_name ./data/train_dataset \
  --output_dir ./outputs/eval_qwen3-1.4b-finetune \
  --do_eval \
  --eval_retrieval \
  --sparse_retrieval_type bm25 \
  --top_k_retrieval 10
```

## 새로운 인자 설명

### `--prompt_template_type`
- **기본값**: `instruction`
- **옵션**: `simple`, `instruction`, `chat`
- **설명**: 프롬프트 템플릿 형식 선택

### `--max_answer_tokens`
- **기본값**: `50`
- **설명**: 생성할 답변의 최대 토큰 수

## 주의사항

1. **추론 코드 미구현**: `inference.py`는 아직 BERT QA 방식으로 되어 있습니다. 추론이 필요한 경우 별도로 구현이 필요합니다.

2. **메모리 사용량**: CasualLM은 BERT QA보다 메모리를 더 많이 사용할 수 있습니다. 배치 크기를 조정하세요.

3. **EM 평가**: 답변만 생성되도록 학습되었지만, 추론 시 정확한 답변 추출 로직이 필요합니다.

## 다음 단계

- [ ] 추론 코드 구현 (`inference.py` 수정)
- [ ] 생성된 답변에서 정확한 답변 추출 로직 구현
- [ ] EM/F1 평가 메트릭 통합
- [ ] 하이퍼파라미터 튜닝
