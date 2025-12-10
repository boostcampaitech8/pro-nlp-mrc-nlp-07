"""
Simplified LLM inference script for CasualLM Reader.
Performs retrieval and generates answers using instruction prompts.
"""

import json
import logging
import os
import sys
from typing import Callable, List, NoReturn

import evaluate
import torch
from arguments import DataTrainingArguments, ModelArguments
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from data_formatter import create_formatter
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("Inference parameters %s", training_args)
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=True,
    )

    # Retrieval 수행
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(
            tokenizer.tokenize,
            datasets,
            training_args,
            data_args,
        )

    # CasualLM 모델 로드
    if training_args.do_eval or training_args.do_predict:
        print("Loading CasualLM Reader model...")
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        )

        # pad_token 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            config.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )

        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    """Retrieval 수행 (기존 코드와 동일)"""
    
    retrieval_type = data_args.sparse_retrieval_type.lower()

    # Retriever 선택 및 초기화
    if retrieval_type == "bm25":
        from retriever.sparse_bm25 import SparseRetrieval
        print("Using Sparse Retrieval: BM25")
        retriever = SparseRetrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path
        )
        retriever.get_sparse_embedding()
    elif retrieval_type.startswith("dense"):
        if retrieval_type == "dense-qwen":
            from retriever.dense_qwen3_embedding import DenseRetrieval
            print("Using Dense Retrieval: Qwen3-Embedding")
        else:
            raise ValueError(f"Unknown dense retrieval type: {retrieval_type}")
        
        retriever = DenseRetrieval(
            data_path=data_path,
            context_path=context_path
        )
        retriever.get_dense_embedding()
    else:
        from retriever.sparse_tfidf import SparseRetrieval
        print("Using Sparse Retrieval: TF-IDF")
        retriever = SparseRetrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path
        )
        retriever.get_sparse_embedding()

    # Retrieve
    df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # 메모리 정리
    print("Cleaning up retriever from memory...")
    del retriever
    if retrieval_type.startswith("dense"):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # Dataset 생성
    if training_args.do_predict:
        f = Features({
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        })
        # 필요한 컬럼만 선택
        df = df[["id", "question", "context"]]
    else:  # do_eval
        f = Features({
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
        })
        # 필요한 컬럼만 선택
        df = df[["id", "question", "context", "answers"]]

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
    """
    CasualLM을 사용한 MRC 추론.
    Instruction 프롬프트로 답변 생성.
    """

    # Prompt formatter 생성
    formatter = create_formatter(template_type=data_args.prompt_template_type)
    print(f"Using prompt template: {data_args.prompt_template_type}")

    eval_dataset = datasets["validation"]

    # GPU로 모델 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"\nGenerating answers for {len(eval_dataset)} examples...")
    print(f"Device: {device}")

    predictions = []
    references = []

    # 각 예제에 대해 답변 생성
    for idx, example in enumerate(eval_dataset):
        if idx % 100 == 0:
            print(f"Processing {idx}/{len(eval_dataset)}...")

        # Prompt 생성 (답변 제외)
        formatted = formatter.format_example(
            {
                "question": example["question"],
                "context": example["context"],
                "answers": example.get("answers", {"text": [""]}),
            },
            include_answer=False
        )
        prompt = formatted["prompt"]

        # 토큰화
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=data_args.max_seq_length,
            truncation=True,
        ).to(device)


        # 생성 - 짧은 답변 유도
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # 최대 20토큰 (약 10단어)
                temperature=0.1,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 디버그: 첫 10개 예제만 출력
        if idx < 10:
            print(f"\n=== Example {idx} ===")
            print(f"Generated text: {generated_text[:300]}...")
        
        # 답변 추출
        answer = ""
        
        # [ANSWER] 태그 이후 추출
        if "[ANSWER]" in generated_text:
            answer = generated_text.split("[ANSWER]")[-1].strip()
        else:
            # 프롬프트 제거
            answer = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()
        
        # 첫 번째 문장만 추출 (마침표, 줄바꿈 기준)
        if answer:
            # 줄바꿈으로 분리
            lines = answer.split('\n')
            answer = lines[0].strip()
            
            # 마침표로 분리
            sentences = answer.split('.')
            answer = sentences[0].strip()
            
            # 쉼표로 분리 (첫 번째 구문만)
            if ',' in answer:
                answer = answer.split(',')[0].strip()
        
        # 정리
        answer = " ".join(answer.split())
        
        # 최대 길이 제한 (10단어 = 약 30자)
        if len(answer) > 30:
            # 공백 기준으로 단어 분리
            words = answer.split()
            if len(words) > 10:
                answer = " ".join(words[:10])
            else:
                answer = answer[:30]
        
        if idx < 10:
            print(f"Final answer: '{answer}'")


        # 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 디버그: 첫 10개 예제만 출력
        if idx < 10:
            print(f"\n=== Example {idx} ===")
            print(f"Prompt length: {len(prompt)}")
            print(f"Generated text: {generated_text[:500]}...")
        
        # 답변 추출 - 여러 방법 시도
        answer = ""
        
        # 방법 1: [ANSWER] 태그 찾기
        if "[ANSWER]" in generated_text:
            answer = generated_text.split("[ANSWER]")[-1].strip()
            if idx < 10:
                print(f"Method 1 (tag): {answer[:100]}")
        
        # 방법 2: 프롬프트 제거
        if not answer:
            # 생성된 텍스트에서 프롬프트 부분 제거
            if generated_text.startswith(prompt):
                answer = generated_text[len(prompt):].strip()
            else:
                # 프롬프트가 정확히 일치하지 않으면 대략적으로 제거
                answer = generated_text.strip()
            
            if idx < 10:
                print(f"Method 2 (remove prompt): {answer[:100]}")
        
        # 방법 3: 줄바꿈 기준으로 마지막 부분 추출
        if not answer or len(answer) > 200:  # 너무 길면 잘못 추출된 것
            lines = generated_text.strip().split('\n')
            # 마지막 비어있지 않은 줄 찾기
            for line in reversed(lines):
                if line.strip() and not line.strip().startswith('['):
                    answer = line.strip()
                    break
            
            if idx < 10:
                print(f"Method 3 (last line): {answer[:100]}")
        
        # 정리: 불필요한 공백, 줄바꿈 제거
        answer = " ".join(answer.split())
        
        # 너무 길면 자르기 (최대 100자)
        if len(answer) > 100:
            answer = answer[:100]
        
        if idx < 10:
            print(f"Final answer: {answer}")


        predictions.append({
            "id": example["id"],
            "prediction_text": answer
        })

        # 참조 답변 저장 (평가용)
        if "answers" in example and example["answers"] is not None:
            references.append({
                "id": example["id"],
                "answers": example["answers"]
            })

    # 예측 결과 저장
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    predictions_file = os.path.join(output_dir, "predictions.json")
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\nPredictions saved to: {predictions_file}")

    # 평가
    if training_args.do_eval and references:
        print("\nEvaluating predictions...")

        metric = evaluate.load("squad")
        results = metric.compute(predictions=predictions, references=references)

        print(f"\nResults:")
        print(f"  Exact Match: {results['exact_match']:.2f}")
        print(f"  F1 Score: {results['f1']:.2f}")

        metrics_file = os.path.join(output_dir, "eval_results.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Metrics saved to: {metrics_file}")

    elif training_args.do_predict:
        print("\nNo metric can be presented because there is no correct answer given. Job done!")


if __name__ == "__main__":
    main()
