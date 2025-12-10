import os
import pandas as pd
from datasets import load_from_disk, Dataset, DatasetDict, Features, Value
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, HfArgumentParser, TrainingArguments
from arguments import ModelArguments, DataTrainingArguments
from inference import run_mrc  # 기존 inference.py의 함수 재사용
from retrieval_bm25 import BM25Retrieval # 아까 만든 BM25 클래스 import

def main():
    # 1. 설정 및 파라미터 로드
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Data: {data_args.dataset_name}")
    print(f"Retrieval Top-k: {data_args.top_k_retrieval}")

    # 2. 모델 & 토크나이저 로드 (Reader용)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path)
    
    # 3. 데이터셋 로드
    datasets = load_from_disk(data_args.dataset_name)

    # 4. [핵심] BM25 Retrieval 수행 (Rerank 없음)
    print("Initializing BM25 Retriever...")
    retriever = BM25Retrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path="../data",
        context_path="wikipedia_documents.json"
    )
    retriever.get_bm25() # 인덱스 생성 또는 로드
    
    print(f"Retrieving Top-{data_args.top_k_retrieval} using Pure BM25...")
    # 순수 BM25 점수대로 상위 k개 문서를 가져와서 합칩니다.
    df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)
    
    # 5. 데이터셋 포맷 변환 (DataFrame -> Dataset)
    if training_args.do_predict:
        # Test dataset 구조에 맞춤
        f = Features({
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        })
    else:
        # Validation dataset 구조 (정답 포함)
        f = datasets["validation"].features
        
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    
    # 6. Reader 추론 실행 (기존 함수 활용)
    print("Starting Reader Inference...")
    run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

if __name__ == "__main__":
    main()