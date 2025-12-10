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
from data_formatter import create_formatter
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)


seed = 2024
deterministic = False

random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)
if deterministic: # cudnn random seed 고정 - 고정 시 학습 속도가 느려질 수 있습니다. 
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig와 Tokenizer를 먼저 로드합니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        use_fast=True,
    )
    
    # CasualLM에서는 pad_token이 필요합니다
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.eos_token_id
    
    # AutoModelForCausalLM을 사용하여 생성 모델을 로드합니다
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:
    """
    Run MRC training/evaluation using CasualLM approach.
    Converts QA data to instruction-based prompts and trains with answer-only loss.
    """
    
    # Create prompt formatter
    formatter = create_formatter(template_type=data_args.prompt_template_type)
    print(f"Using prompt template: {data_args.prompt_template_type}")
    
    # Prepare training dataset - preprocess once before training
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        
        print("Preprocessing training dataset...")
        train_dataset = datasets["train"]
        
        # Format and tokenize examples
        def preprocess_function(examples):
            """Convert QA examples to tokenized format with answer-only labels."""
            # Format prompts
            formatted = formatter.format_batch(examples, include_answer=True)
            prompts = formatted["prompt"]
            full_texts = formatted["full_text"]
            
            # Tokenize full texts
            model_inputs = tokenizer(
                full_texts,
                max_length=data_args.max_seq_length,
                truncation=True,
                padding=False,  # Will pad in data collator
            )
            
            # Tokenize prompts to find answer positions
            prompt_inputs = tokenizer(
                prompts,
                max_length=data_args.max_seq_length,
                truncation=True,
                padding=False,
            )
            
            # Create labels with prompt tokens masked
            labels = []
            for idx in range(len(full_texts)):
                label = model_inputs["input_ids"][idx].copy()
                prompt_length = len(prompt_inputs["input_ids"][idx])
                
                # Mask prompt tokens with -100
                for i in range(min(prompt_length, len(label))):
                    label[i] = -100
                
                labels.append(label)
            
            model_inputs["labels"] = labels
            return model_inputs
        
        
        # Apply preprocessing
        preprocessed_train_path = os.path.join(data_args.dataset_name, "preprocessed_train")
        
        if os.path.exists(preprocessed_train_path) and not data_args.overwrite_cache:
            print(f"Loading preprocessed training data from {preprocessed_train_path}")
            train_dataset = load_from_disk(preprocessed_train_path)
        else:
            print("Preprocessing training data...")
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=False,
                desc="Tokenizing training data",
            )
            print(f"Saving preprocessed training data to {preprocessed_train_path}")
            train_dataset.save_to_disk(preprocessed_train_path)
        
        
        print(f"Training dataset size: {len(train_dataset)}")
    
    # Prepare evaluation dataset - preprocess once before evaluation
    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        
        print("Preprocessing evaluation dataset...")
        eval_dataset = datasets["validation"]
        
        
        # Apply same preprocessing
        preprocessed_eval_path = os.path.join(data_args.dataset_name, "preprocessed_validation")
        
        if os.path.exists(preprocessed_eval_path) and not data_args.overwrite_cache:
            print(f"Loading preprocessed evaluation data from {preprocessed_eval_path}")
            eval_dataset = load_from_disk(preprocessed_eval_path)
        else:
            print("Preprocessing evaluation data...")
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=False,
                desc="Tokenizing evaluation data",
            )
            print(f"Saving preprocessed evaluation data to {preprocessed_eval_path}")
            eval_dataset.save_to_disk(preprocessed_eval_path)
        
        
        print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    
    # Use custom data collator for padding preprocessed data
    from data_collator import DataCollatorForPreprocessedLM
    data_collator = DataCollatorForPreprocessedLM(
        tokenizer=tokenizer,
    )
    
    print("\nData collator configured for padding only")
    print(f"Max sequence length: {data_args.max_seq_length}")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    if training_args.do_train:
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50)
        
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        
        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")
        
        # Save trainer state
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )
        
        print("\nTraining completed!")
        print(f"Model saved to: {training_args.output_dir}")
    
    # Evaluation
    if training_args.do_eval:
        print("\n" + "="*50)
        print("Starting evaluation...")
        print("="*50)
        
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        print("\nEvaluation completed!")
        print(f"Evaluation loss: {metrics.get('eval_loss', 'N/A')}")



if __name__ == "__main__":
    main()
