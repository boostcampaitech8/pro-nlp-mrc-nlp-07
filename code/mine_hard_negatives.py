# mine_hard_negatives.py

import json
import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

# --- 1. 설정 및 경로 ---
SEED = 42
DATA_PATH = "../data"
WIKI_CONTEXT_PATH = os.path.join(DATA_PATH, "wikipedia_documents.json")
TRAIN_DATASET_PATH = os.path.join(DATA_PATH, "train_dataset")
BM25_MODEL_NAME = "klue/roberta-large" # BM25 토크나이징에 사용될 토크나이저
DENSE_MODEL_NAME = "BAAI/bge-m3"
TOP_K_RETRIEVAL = 50 # 각 모델당 Top 50 문맥을 검색하여 Negative Sample 후보로 사용
OUTPUT_FILENAME = "train_with_hard_negatives.json"
NUM_NEGATIVES_PER_QUERY = 5 # 최종적으로 추출할 Negative Sample의 개수 (BM25 + Dense)

# 시드 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 유틸리티 클래스: BM25 ---
class BM25MinimalRetriever:
    def __init__(self, contexts: List[str], tokenize_fn: callable):
        self.contexts = contexts
        self.tokenize_fn = tokenize_fn
        
        print("Tokenizing contexts for BM25...")
        self.tokenized_contexts = [self.tokenize_fn(doc) for doc in tqdm(self.contexts, desc="Tokenizing")]
        
        print("Building BM25 model...")
        self.bm25_model = BM25Okapi(self.tokenized_contexts)

    def retrieve_single(self, query: str, k: int) -> List[int]:
        """단일 쿼리에 대한 BM25 검색 및 인덱스 반환"""
        tokenized_query = self.tokenize_fn(query)
        doc_scores = self.bm25_model.get_scores(tokenized_query)
        sorted_result = np.argsort(doc_scores)[::-1]
        return sorted_result.tolist()[:k]

# Dense Retriever 모델 로드
def get_dense_model(model_name: str) -> SentenceTransformer:
    print(f"Loading Dense Model: {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.max_seq_length = 512
    model.to(DEVICE)
    model.eval()
    return model

# --- 3. Hard Negative Mining 메인 함수 ---
def mine_negatives():
    
    # 3.1. 데이터 및 문맥 로드
    print("Loading Data...")
    
    # Corpus (Passage) 로드
    with open(WIKI_CONTEXT_PATH, "r", encoding="utf-8") as f:
        wiki = json.load(f)
    contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
    context_to_id = {text: i for i, text in enumerate(contexts)}
    print(f"Total Unique Contexts: {len(contexts)}")

    # Train Query 데이터 로드
    org_dataset = load_from_disk(TRAIN_DATASET_PATH)
    train_ds = org_dataset["train"].flatten_indices() 
    queries = train_ds["question"]
    ground_truth_contexts = train_ds["context"]
    
    # Ground Truth Passage의 인덱스(ID)를 찾습니다.
    # get(..., -1)을 사용하여 문맥이 wikipedia_documents.json에 없으면 -1을 할당 (오류 방지)
    ground_truth_pids = [context_to_id.get(ctx, -1) for ctx in ground_truth_contexts]

    # 3.2. BM25 Hard Negative 추출
    print("\n[Stage 1/3] BM25 Hard Negative Mining...")
    tokenizer = AutoTokenizer.from_pretrained(BM25_MODEL_NAME)
    bm25_retriever = BM25MinimalRetriever(contexts, tokenize_fn=tokenizer.tokenize)
    
    bm25_hn_indices = []
    for i, query in enumerate(tqdm(queries, desc="BM25 Searching")):
        top_k_indices = bm25_retriever.retrieve_single(query, k=TOP_K_RETRIEVAL)
        
        # 정답 문맥(P+)과 겹치지 않고, ID가 유효한 인덱스만 필터링
        hn_list = [pid for pid in top_k_indices if pid != ground_truth_pids[i] and pid != -1]
        bm25_hn_indices.append(hn_list)

    # 메모리 정리
    del bm25_retriever
    del tokenizer

    # 3.3. Dense Hard Negative 추출 (BAAI/bge-m3)
    # print("\n[Stage 2/3] Dense Hard Negative Mining...")
    
    # dense_model = get_dense_model(DENSE_MODEL_NAME)
    
    # # Passage 임베딩 
    # print("Encoding Contexts...")
    # corpus_embeddings = dense_model.encode(
    #     contexts, 
    #     batch_size=256, 
    #     show_progress_bar=True, 
    #     convert_to_tensor=True
    # ).to(DEVICE)

    # # Query 임베딩
    # print("Encoding Queries...")
    # bge_queries = [f"query: {q}" for q in queries] 
    # query_embeddings = dense_model.encode(
    #     bge_queries, 
    #     batch_size=256, 
    #     show_progress_bar=True, 
    #     convert_to_tensor=True
    # ).to(DEVICE)
    
    # # 유사도 검색 (util.semantic_search는 내부적으로 코사인 유사도와 FAISS/torch matrix 연산을 활용)
    # print("Semantic Search (Dense)...")
    # hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=TOP_K_RETRIEVAL)
    
    # dense_hn_indices = []
    # for i, hit_list in enumerate(tqdm(hits, desc="Dense Searching")):
    #     top_k_indices = [hit['corpus_id'] for hit in hit_list]
        
    #     # 정답 문맥(P+)과 겹치지 않고, ID가 유효한 인덱스만 필터링
    #     hn_list = [pid for pid in top_k_indices if pid != ground_truth_pids[i] and pid != -1]
    #     dense_hn_indices.append(hn_list)

    # # 메모리 정리
    # del dense_model
    # del corpus_embeddings
    # del query_embeddings
    # torch.cuda.empty_cache()

    # 3.4. 최종 데이터셋 구성 및 저장
    print("\n[Stage 3/3] Finalizing Dataset...")
    
    final_examples = []

    for i in tqdm(range(len(queries)), desc="Combining Negatives"):
        
        positive_context = ground_truth_contexts[i]
        
        # Hard Negative Pool (BM25 + Dense 합치고 중복 제거)
        hn_pool_indices = set(bm25_hn_indices[i])
        hn_pool_indices.discard(ground_truth_pids[i]) # 다시 한번 정답 제거
        
        # Pool에서 필요한 개수만큼 랜덤 샘플링
        selected_hn_indices = random.sample(
            list(hn_pool_indices), 
            min(NUM_NEGATIVES_PER_QUERY, len(hn_pool_indices)) 
        )
        
        # Negative Context 텍스트 추출
        negative_contexts = [contexts[pid] for pid in selected_hn_indices]
        
        # 최종 JSON 형식으로 저장
        final_examples.append({
            "id": train_ds["id"][i],
            "question": queries[i],
            "positive_context": positive_context,
            "negative_contexts": negative_contexts, # Negative 문맥은 리스트 형태로 저장
        })

    # JSON 파일로 저장
    output_path = os.path.join(DATA_PATH, OUTPUT_FILENAME)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_examples, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Hard Negative Mining Complete. Saved to: {output_path}")

if __name__ == "__main__":
    # 이 스크립트를 실행하기 전에 'pip install sentence-transformers rank-bm25 transformers datasets' 필요
    mine_negatives()