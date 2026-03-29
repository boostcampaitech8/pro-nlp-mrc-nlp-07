"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.
(Hybrid Retrieval 캐싱 및 토크나이저 충돌 방지 로직 포함)
"""
import logging
import os
import sys
import json
import pandas as pd # pandas 추가
from typing import Callable, Dict, List, NoReturn, Tuple

import evaluate
import numpy as np
from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
)
# [수정] 파일 이름에 맞게 import 합니다.
from retriever_hybrid import HybridRetrieval 
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions

logger = logging.getLogger(__name__)

# Alpha 값을 하드코딩하지 않고, 실행 시 인수로 받을 수 있도록 DataTrainingArguments에 추가하거나
# 임시로 이 부분에 고정합니다. (현재는 0.3으로 고정)
OPTIMAL_ALPHA = 0.3 


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 이 파일은 평가 전용입니다.
    training_args.do_train = False 

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name) 

    # [수정] local_files_only=True 추가하여 로컬 모델 로드 오류 방지
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        local_files_only=True
    )
    # [수정] local_files_only=True 추가
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        use_fast=True,
        local_files_only=True
    )
    # [수정] local_files_only=True 추가
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        local_files_only=True
    )

    # True일 경우 : run passage retrieval (Hybrid)
    if data_args.eval_retrieval:
        datasets = run_hybrid_retrieval(
            datasets,
            training_args,
            data_args,
        )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_hybrid_retrieval(
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    
    # --------------------------------------------------------------------------
    # [캐싱 로직 시작]
    # 캐시 파일 경로 정의: output_dir에 alpha와 top_k를 포함한 파일명으로 저장
    global OPTIMAL_ALPHA
    
    CACHE_FILENAME = f"hybrid_results_a{OPTIMAL_ALPHA}_k{data_args.top_k_retrieval}.json"
    CACHE_PATH = os.path.join(training_args.output_dir, CACHE_FILENAME)

    if os.path.exists(CACHE_PATH):
        print(f"✅ Loading cached hybrid retrieval results from: {CACHE_PATH}")
        # json lines 형식으로 저장된 파일을 로드
        df = pd.read_json(CACHE_PATH, orient='records', lines=True)
    else:
        print("🚀 Starting Hybrid Retrieval and Caching results...")

        # 1. Hybrid Retriever 초기화
        retriever = HybridRetrieval(
            dataset_path=data_args.dataset_name,
            context_path=os.path.join(data_path, context_path),
            bm25_tokenizer_name="klue/roberta-large"
        )
        
        # 2. Hybrid 검색 수행 (Retrieve & Re-rank)
        is_eval_run = training_args.do_eval 

        print(f"Performing Hybrid Retrieval with Optimal Alpha={OPTIMAL_ALPHA}")

        # retriever_hybrid.py의 retrieve_hybrid 함수 호출
        df = retriever.retrieve_hybrid(
            top_k=data_args.top_k_retrieval, 
            alpha=OPTIMAL_ALPHA,
            is_eval=is_eval_run
        )

        # 3. 검색 결과 저장 (캐시 파일 생성)
        os.makedirs(training_args.output_dir, exist_ok=True)
        # DataFrame을 JSON Lines 형식으로 저장
        df.to_json(CACHE_PATH, orient='records', lines=True)
        print(f"💾 Retrieval results cached to: {CACHE_PATH}")
    # [캐싱 로직 끝]
    # --------------------------------------------------------------------------

    # 4. 데이터셋 형식 변환 (df가 캐시에서 로드되었거나 새로 생성되었거나)
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
        if 'answers' in df.columns:
            df = df.drop(columns=['answers'])

    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    # [중요] 검색된 context로 datasets의 validation 셋을 덮어씁니다.
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else None

    # Padding에 대한 옵션을 설정합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Validation preprocessing
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
        
        # ----------------------------------------------------------------------
        # [최종 수정] 토크나이저 충돌 방지: token_type_ids를 명시적으로 제거합니다.
        if "token_type_ids" in tokenized_examples:
             del tokenized_examples["token_type_ids"]
        # ----------------------------------------------------------------------

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

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            
            if answer_column_name:
                references = [
                    {"id": ex["id"], "answers": ex[answer_column_name]}
                    for ex in datasets["validation"]
                ]
            else:
                logger.warning("Answers column not found in dataset for evaluation. Skipping metric calculation.")
                return EvalPrediction(predictions=formatted_predictions, label_ids=None)

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        if p.label_ids is None:
            return {"f1": 0.0, "exact_match": 0.0}
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )
        print(
            "Prediction file saved. No metric can be presented because there is no correct answer given."
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()