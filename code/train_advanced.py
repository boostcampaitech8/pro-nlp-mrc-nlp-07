import logging
import os
import sys
from typing import NoReturn

import torch
from arguments import DataTrainingArguments, ModelArguments
import evaluate
from datasets import load_from_disk 
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions

logger = logging.getLogger(__name__)

def main():
    # [기존 로직 유지] 인자 파싱
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # [수정] CLI 인자 의존성을 제거하고 학습 안정성을 위해 파라미터 강제 할당
    # 의도: 명령어 실수로 인한 설정 오류(save/eval strategy 불일치 등) 방지 및 과적합 규제 적용
    print("설정을 코드 내부에서 강제로 적용합니다 (CLI 인자 무시)")
    
    # 학습 및 평가 전략 동기화
    training_args.do_train = True
    training_args.do_eval = True                     
    training_args.evaluation_strategy = "epoch"      # 에포크마다 평가 수행
    training_args.save_strategy = "epoch"            # 에포크마다 저장 (평가 전략과 일치 필수)
    training_args.load_best_model_at_end = True      # 학습 종료 시 EM 점수가 가장 높은 모델 로드
    training_args.metric_for_best_model = "exact_match"
    training_args.greater_is_better = True
    training_args.save_total_limit = 1               # 용량 최적화를 위해 베스트 모델 1개만 유지
    
    # 과적합 방지를 위한 규제 설정
    training_args.label_smoothing_factor = 0.1       # 정답에 대한 과신 방지
    training_args.lr_scheduler_type = "cosine"       # 학습률을 부드럽게 감소
    training_args.warmup_ratio = 0.1                 # 초기 학습 안정화
    
    # [기존 로직 유지] 로깅 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # [기존 로직 유지] 시드 고정
    set_seed(training_args.seed)

    # [기존 로직 유지] 데이터셋 로드
    datasets = load_from_disk(data_args.dataset_name)

    # [추가] Validation 데이터셋 부재 시 안전장치
    # 의도: 데이터 병합 과정에서 검증셋이 누락되었을 경우 Train의 10%를 검증용으로 자동 분리
    if "validation" not in datasets:
        print("Validation 데이터가 확인되지 않아 Train의 10%를 검증용으로 분리합니다.")
        split_datasets = datasets["train"].train_test_split(test_size=0.1, seed=training_args.seed)
        datasets["train"] = split_datasets["train"]
        datasets["validation"] = split_datasets["test"]

    # [기존 로직 유지] 모델 및 토크나이저 로드
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=True
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # [기존 로직 유지] 전처리 설정 (컬럼명 매핑 및 패딩 방향 확인)
    column_names = datasets["train"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    pad_on_right = tokenizer.padding_side == "right"

    # [기존 로직 유지] 오류 체크
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # [기존 로직 유지] 학습 데이터 전처리 함수
    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False, 
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    # [기존 로직 유지] 학습 데이터셋 매핑
    train_dataset = datasets["train"]
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # [기존 로직 유지] 검증 데이터 전처리 (Best Model 선정을 위해 필요)
    eval_dataset = datasets["validation"]
    
    def prepare_validation_features(examples):
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_features = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # [기존 로직 유지] Data Collator
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # [기존 로직 유지] Metric 로드
    metric = evaluate.load("squad")

    # [수정] compute_metrics 함수 재정의
    # 의도: Trainer 내부에서 매 에포크마다 평가를 수행할 때, 로그에 EM/F1 점수를 명확히 출력하기 위함
    # 기존 코드에서는 compute_metrics=None 이었으나, 실시간 모니터링을 위해 구현함
    def compute_metrics(p):
        predictions = [x[0] for x in p]
        references = [x[1] for x in p]
        
        results = metric.compute(predictions=predictions, references=references)
        
        # 터미널 출력용 포맷팅
        print("\n" + "="*30)
        print(f"EVALUATION RESULT")
        print(f"   Exact Match (EM): {results['exact_match']:.2f}")
        print(f"   F1 Score        : {results['f1']:.2f}")
        print("="*30 + "\n")
        
        logger.info(f"Evaluation metrics: {results}")
        return results

    # [기존 로직 유지] Post Processing 함수
    def post_processing_function(examples, features, predictions, stage="eval"):
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=0.0,
            output_dir=training_args.output_dir,
            prefix=stage,
        )
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return list(zip(formatted_predictions, references))

    # [수정] Trainer 초기화 시 compute_metrics 연결
    # 의도: 위에서 정의한 평가 로직을 Trainer에 연결하여 에포크마다 자동으로 실행되도록 함
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_features,
        eval_examples=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics, # 변경됨: None -> compute_metrics
    )

    # [수정] 학습 실행 흐름 단순화
    # 의도: do_train, do_eval 분기 처리를 제거하고 한 번의 실행으로 학습, 평가, 베스트 모델 저장을 완료함
    print("Training Start with Advanced Techniques")
    trainer.train()
    
    # Best Model 저장
    trainer.save_model()
    print(f"Best Model Saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()