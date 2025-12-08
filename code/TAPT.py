import os
import sys
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from datasets import load_dataset

logger = logging.getLogger(__name__)

@dataclass
class TaptArguments:
    model_name_or_path: str = field(
        default="klue/roberta-large", 
        metadata={"help": "Pretrained model name"}
    )
    dataset_name: str = field(
        default="../data/train_dataset", 
        metadata={"help": "Path to dataset"}
    )
    # [수정 1] output_dir 제거 (TrainingArguments와 충돌 방지)

def main():
    parser = HfArgumentParser((TaptArguments, TrainingArguments))
    tapt_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    set_seed(training_args.seed)

    # 1. 데이터 로드
    logger.info(f"Loading dataset from {tapt_args.dataset_name}")
    
    try:
        # 시도 1: load_from_disk (이미 전처리되어 저장된 데이터셋일 경우)
        from datasets import load_from_disk
        dataset = load_from_disk(tapt_args.dataset_name)
        logger.info("Loaded from disk successfully.")
    except:
        # 시도 2: load_dataset (일반 json 파일이나 스크립트일 경우)
        logger.info("load_from_disk failed. Trying load_dataset...")
        if tapt_args.dataset_name.endswith(".json"):
            dataset = load_dataset("json", data_files={"train": tapt_args.dataset_name})
        else:
            # 폴더지만 json이 아닌 huggingface dataset 구조일 때
            dataset = load_dataset(tapt_args.dataset_name)

    # 2. 모델 & 토크나이저 로드
    config = AutoConfig.from_pretrained(tapt_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tapt_args.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(tapt_args.model_name_or_path, config=config)

    # 3. 전처리: Context만 추출하여 토큰화 (MLM 학습용)
    # 데이터셋 구조 유연성 확보 (split이 없는 경우 대비)
    if "train" not in dataset:
        if isinstance(dataset, dict):
             # split 키가 없다면 첫 번째 키를 사용하거나 전체를 train으로 간주
             dataset = {"train": dataset[list(dataset.keys())[0]]}
    
    column_names = dataset["train"].column_names
    text_column_name = "context" if "context" in column_names else column_names[0]

    def tokenize_function(examples):
        # RoBERTa는 token_type_ids가 필요 없으므로 별도 설정 불필요 (자동 처리)
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True, truncation=True, max_length=512)

    tokenized_datasets = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    # 4. Data Collator (MLM 마스킹 자동 적용)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # 5. Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. 학습 시작
    logger.info("Start TAPT (Masked Language Modeling)...")
    train_result = trainer.train()
    
    # [수정 2] tapt_args.output_dir -> training_args.output_dir 사용
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Perplexity (성능 지표) 계산 및 로그
    metrics = train_result.metrics
    try:
        perplexity = math.exp(metrics["train_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity
    
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()