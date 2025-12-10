# train.py
import logging
import os
import sys
import random
import numpy as np
import torch
import evaluate
from typing import NoReturn

from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk
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


seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ✅ V100 + roberta-large 최적 세팅 강제 반영
    training_args.per_device_train_batch_size = 4
    training_args.gradient_accumulation_steps = 4
    training_args.num_train_epochs = 5
    training_args.learning_rate = 2e-5
    training_args.fp16 = True
    training_args.fp16_full_eval = True
    training_args.warmup_ratio = 0.1
    training_args.weight_decay = 0.01
    training_args.lr_scheduler_type = "cosine"
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = 500
    training_args.save_steps = 500
    training_args.logging_steps = 50
    training_args.save_total_limit = 2

    # ✅ 가장 중요한 BEST MODEL 저장 옵션
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "f1"
    training_args.greater_is_better = True

    data_args.max_seq_length = 384
    data_args.doc_stride = 192
    data_args.max_answer_length = 30
    data_args.pad_to_max_length = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path, config=config)

    run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(data_args, training_args, model_args, datasets, tokenizer, model) -> NoReturn:
    column_names = datasets["train"].column_names if training_args.do_train else datasets["validation"].column_names

    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    pad_on_right = tokenizer.padding_side == "right"

    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

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
            padding=False,
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
                    while offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        train_dataset = datasets["train"].map(
            prepare_train_features,
            batched=True,
            remove_columns=column_names,
        )

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
            padding=False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(datasets["validation"][sample_index]["id"])
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

    eval_dataset = datasets["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=postprocess_qa_predictions,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
