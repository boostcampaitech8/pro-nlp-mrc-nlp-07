import json
import os
import random
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer, util, models

# 시드 고정
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def load_data(dataset_path, context_path):
    """데이터 로드 함수"""
    print(f"Loading data from {dataset_path}...")
    
    # 1. Corpus(Passage) 로드
    with open(context_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    print(f"Unique Contexts: {len(contexts)}")

    # 2. Query(Dataset) 로드
    org_dataset = load_from_disk(dataset_path)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )
    print(f"Total Queries: {len(full_ds)}")
    
    return contexts, full_ds

def get_model(model_name):
    """모델 로드 함수 (Sequence Length 제한 추가)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
    except Exception:
        print(f"Warning: {model_name} 로드 중 에러 발생, Transformer 방식으로 우회 시도...")
        word_embedding_model = models.Transformer(model_name, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # BGE-M3 등이 메모리 폭발하지 않도록 길이 제한 (ODQA에서는 512면 충분)
    model.max_seq_length = 512
    model.to(device)
    return model

def evaluate_dense(model_name, contexts, dataset, top_k=20, batch_size=16):
    """단일 모델 평가 함수"""
    print(f"\n[{model_name}] 평가 시작...")
    
    # 메모리 초기화 (이전 모델 잔여물 제거)
    torch.cuda.empty_cache()
    
    model = get_model(model_name)
    
    # 1. Passage Embedding
    print("Encoding Contexts (Passages)...")
    corpus_embeddings = model.encode(contexts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)

    # 2. Query Embedding
    print("Encoding Queries...")
    queries = dataset["question"]
    
    # E5 계열은 prefix 필요
    if "e5" in model_name.lower():
        queries = [f"query: {q}" for q in queries]
    
    query_embeddings = model.encode(queries, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)

    # 3. Similarity Search
    print("Calculating Similarity...")
    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k)

    # 4. Recall 계산
    correct_count = 0
    total_count = len(dataset)

    for idx, hit in enumerate(hits):
        ground_truth = dataset[idx]["context"]
        retrieved_contexts = [contexts[h['corpus_id']] for h in hit]
        
        if ground_truth in retrieved_contexts:
            correct_count += 1

    recall = correct_count / total_count
    print(f"Result >> [{model_name}] Recall@{top_k}: {recall:.4f}")
    
    # 메모리 해제
    del model
    del corpus_embeddings
    del query_embeddings
    torch.cuda.empty_cache()
    
    return recall

if __name__ == "__main__":
    # 경로 설정
    DATASET_PATH = "../data/train_dataset"
    CONTEXT_PATH = "../data/wikipedia_documents.json"
    
    # 비교할 모델 리스트
    models_to_compare = [
        "intfloat/multilingual-e5-large-instruct", 
        "BAAI/bge-m3",
        "Alibaba-NLP/gte-multilingual-base",
        "BM-K/KoSimCSE-roberta-multitask",
    ]
    
    contexts, dataset = load_data(DATASET_PATH, CONTEXT_PATH)
    
    results = {}
    
    for model_name in models_to_compare:
        # 안전빵을 위해 batch_size 16으로 고정
        score = evaluate_dense(model_name, contexts, dataset, top_k=20, batch_size=16)
        results[model_name] = score

    print("\n" + "="*50)
    print("🏆 Final Leaderboard (Recall@20) 🏆")
    print("="*50)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(sorted_results, 1):
        print(f"{rank}위: {name} | {score:.4f}")