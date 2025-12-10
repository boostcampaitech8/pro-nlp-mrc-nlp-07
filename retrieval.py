
import json
import os
import time
import random
import pickle
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

# Elasticsearch
from elasticsearch import Elasticsearch, helpers

# Deep Learning (DPR + Reranker)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class ElasticSearchRetrieval:
    def __init__(
        self,
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
        es_index_name: str = "wiki-index",
        
        rerank_model_path: str = "Dongjin-kr/ko-reranker", 
        dense_model_name: str = "jhgan/ko-sroberta-multitask", # 한국어 성능 좋은 Dense 모델
    ) -> NoReturn:
        
        self.data_path = data_path
        self.context_path = context_path
        self.es_index_name = es_index_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Elasticsearch 설정
        self.es = Elasticsearch("http://localhost:9200", timeout=30, max_retries=10, retry_on_timeout=True)
        
        # 2. Context 데이터 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(self.contexts)}")

        # 3. Elasticsearch 인덱싱
        if not self.es.indices.exists(index=self.es_index_name):
            print("Creating and Indexing Elasticsearch...")
            self._setup_es_index()
            self._index_data()
        else:
            print("Elasticsearch index already exists.")

        # 4. [NEW] Dense Retrieval (DPR) 설정
        print(f"Loading Dense Retriever: {dense_model_name}...")
        self.dense_model = SentenceTransformer(dense_model_name, device=self.device)
        self.dense_embedding_path = os.path.join(data_path, "dense_embeddings.bin")
        
        # 임베딩 캐싱 (최초 1회 생성 후 저장)
        if os.path.isfile(self.dense_embedding_path):
            print("Loading Dense Embeddings...")
            with open(self.dense_embedding_path, "rb") as f:
                self.doc_embeddings = pickle.load(f)
        else:
            print("Building Dense Embeddings (First Run - This takes time)...")
            self.doc_embeddings = self.dense_model.encode(
                self.contexts, 
                show_progress_bar=True, 
                convert_to_tensor=True,
                device=self.device
            )
            # CPU로 저장 (GPU 메모리 절약)
            with open(self.dense_embedding_path, "wb") as f:
                pickle.dump(self.doc_embeddings.cpu(), f)
            
        # 검색 시 GPU로 이동
        if torch.cuda.is_available():
            self.doc_embeddings = self.doc_embeddings.to(self.device)

        # 5. Reranker 모델 로드 (학습 없이 바로 로드)
        print(f"Loading Reranker model: {rerank_model_path}...")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_path).to(self.device)
        self.rerank_model.eval()

    def _setup_es_index(self):
        """ES 인덱스 설정 (Nori)"""
        settings = {
            "analysis": {
                "analyzer": {
                    "nori_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "decompound_mode": "mixed",
                    }
                }
            }
        }
        mappings = {"properties": {"content": {"type": "text", "analyzer": "nori_analyzer"}}}
        self.es.indices.create(index=self.es_index_name, body={"settings": settings, "mappings": mappings})

    def _index_data(self):
        """ES 벌크 인덱싱"""
        buffer = []
        for i, context in enumerate(tqdm(self.contexts, desc="Indexing to ES")):
            doc = {"_index": self.es_index_name, "_id": i, "content": context}
            buffer.append(doc)
            if len(buffer) >= 1000:
                helpers.bulk(self.es, buffer)
                buffer = []
        if buffer: helpers.bulk(self.es, buffer)

    def search_es(self, query: str, topk: int = 60) -> List[int]:
        """Sparse Search (Elasticsearch)"""
        res = self.es.search(
            index=self.es_index_name,
            body={"query": {"match": {"content": query}}, "size": topk}
        )
        return [int(hit["_id"]) for hit in res["hits"]["hits"]]

    def search_dense(self, query: str, topk: int = 60) -> List[int]:
        """Dense Search (DPR)"""
        # 쿼리 인코딩
        query_embedding = self.dense_model.encode(query, convert_to_tensor=True, device=self.device)
        # 코사인 유사도 검색
        hits = util.semantic_search(query_embedding, self.doc_embeddings, top_k=topk)[0]
        return [hit['corpus_id'] for hit in hits]

    def rerank(self, query: str, candidate_indices: List[int], topk: int = 5) -> Tuple[List[float], List[int]]:
        """Reranking (Dongjin-kr/ko-reranker 사용)"""
        candidate_contexts = [self.contexts[i] for i in candidate_indices]
        inputs = [(query, ctx) for ctx in candidate_contexts]
        
        encoded_inputs = self.rerank_tokenizer(
            inputs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.rerank_model(**encoded_inputs)
            if outputs.logits.shape[1] == 1:
                scores = outputs.logits.squeeze().tolist()
            else:
                scores = outputs.logits[:, 1].tolist()

        if not isinstance(scores, list): scores = [scores]

        combined = sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)
        return [score for idx, score in combined[:topk]], [idx for idx, score in combined[:topk]]

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 5
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        
        # 1차 검색 후보군 (ES 50개 + DPR 50개)
        # Reranker가 처리할 수 있는 범위 내에서 조절 (너무 많으면 느려짐)
        RETRIEVAL_TOPK = 50 

        if isinstance(query_or_dataset, str):
            query = query_or_dataset
            
            # 1. Hybrid Search
            es_indices = self.search_es(query, topk=RETRIEVAL_TOPK)
            dpr_indices = self.search_dense(query, topk=RETRIEVAL_TOPK)
            candidate_indices = list(set(es_indices) | set(dpr_indices))
            
            # 2. Rerank
            rr_scores, rr_indices = self.rerank(query, candidate_indices, topk=topk)
            
            print(f"[Search query]: {query}")
            for i in range(len(rr_indices)):
                print(f"Top-{i+1} passage score: {rr_scores[i]:.4f}")
                print(self.contexts[rr_indices[i]][:100] + "...")

            return (rr_scores, [self.contexts[i] for i in rr_indices])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            
            print(f"Running Hybrid Retrieval (ES + DPR) -> Rerank")
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Retrieving")):
                query = example["question"]
                
                # 1. Sparse Search
                es_indices = self.search_es(query, topk=RETRIEVAL_TOPK)
                
                # 2. Dense Search
                dpr_indices = self.search_dense(query, topk=RETRIEVAL_TOPK)
                
                # 3. Combine Candidates (합집합)
                candidate_indices = list(set(es_indices) | set(dpr_indices))
                
                # 4. Rerank
                if not candidate_indices:
                    rr_indices = []
                else:
                    _, rr_indices = self.rerank(query, candidate_indices, topk=topk)
                
                tmp = {
                    "question": query,
                    "id": example["id"],
                    "context": " ".join([self.contexts[pid] for pid in rr_indices]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

if __name__ == "__main__":
    # Test Code
    retriever = ElasticSearchRetrieval()
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    retriever.retrieve(query, topk=5)