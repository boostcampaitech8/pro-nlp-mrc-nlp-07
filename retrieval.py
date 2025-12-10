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

# RAG Utilities
from utils_rag import RecursiveTextSplitter, TorchVectorStore

seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class HybridRetrieval:
    def __init__(
        self,
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
        es_index_name: str = "wiki-rag-index", 
        
        rerank_model_path: str = "Qwen/Qwen3-Reranker-0.6B", 
        dense_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> NoReturn:
        
        self.data_path = data_path
        self.context_path = context_path
        self.es_index_name = es_index_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 0. Text Splitter 설정
        self.text_splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # 1. Elasticsearch 설정
        self.es = Elasticsearch("http://localhost:9200")
        
        # 2. Context 데이터 로드 및 청킹 (Chunking)
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        
        # 원본 문서는 유지 (나중에 매핑이 필요할 수 있음) but 지금은 search 결과로 chunk를 반환
        unique_contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        
        print("Splitting documents into chunks...")
        self.contexts = [] # This will now store CHUNKS
        for context in tqdm(unique_contexts, desc="Chunking"):
             chunks = self.text_splitter.split_text(context)
             self.contexts.extend(chunks)
             
        print(f"Total chunks: {len(self.contexts)}")

        # 3. Elasticsearch 인덱싱 (Chunks)
        if not self.es.indices.exists(index=self.es_index_name):
            print("Creating and Indexing Elasticsearch with Chunks...")
            self._setup_es_index()
            self._index_data()
        else:
            print("Elasticsearch index already exists.")

        # 4. Dense Retrieval (DPR) 설정 w/ PyTorch
        print(f"Loading Dense Retriever: {dense_model_name}...")
        self.dense_model = SentenceTransformer(dense_model_name, device=self.device)
        self.dense_index_path = os.path.join(data_path, "torch_dense_index.bin")
        
        # Torch Store 초기화
        self.vector_store = TorchVectorStore(device="cuda" if torch.cuda.is_available() else "cpu")

        if os.path.isfile(self.dense_index_path):
            print("Loading Torch Index...")
            self.vector_store.load(self.dense_index_path)
        else:
            print("Building Dense Embeddings (This takes time)...")
            
            # SentenceTransformer encode
            embeddings = self.dense_model.encode(
                self.contexts, 
                show_progress_bar=True, 
                convert_to_numpy=False, # Use Tensor directly
                convert_to_tensor=True,
                device=self.device
            )
            
            self.vector_store.add_embeddings(embeddings)
            self.vector_store.save(self.dense_index_path)
            
        # 5. Reranker 모델 로드
        print(f"Loading Reranker model: {rerank_model_path}...")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)
        if self.rerank_tokenizer.pad_token is None:
            self.rerank_tokenizer.pad_token = self.rerank_tokenizer.eos_token
        
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_path).to(self.device)
        self.rerank_model.config.pad_token_id = self.rerank_tokenizer.pad_token_id
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
        """Dense Search (DPR + Torch)"""
        # 쿼리 인코딩
        query_embedding = self.dense_model.encode(query, convert_to_tensor=True, device=self.device)
        
        # Torch 검색
        _, indices = self.vector_store.search(query_embedding, topk=topk)
        return indices

    def rerank(self, query: str, candidate_indices: List[int], topk: int = 5) -> Tuple[List[float], List[int]]:
        """Reranking"""
        # 인덱스 유효성 검사
        valid_indices = [i for i in candidate_indices if 0 <= i < len(self.contexts)]
        
        candidate_contexts = [self.contexts[i] for i in valid_indices]
        inputs = [(query, ctx) for ctx in candidate_contexts]
        
        if not inputs:
            return [], []

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

        combined = sorted(zip(valid_indices, scores), key=lambda x: x[1], reverse=True)
        return [score for idx, score in combined[:topk]], [idx for idx, score in combined[:topk]]

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 5
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        
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
            
            print(f"Running Hybrid Retrieval (ES + Torch) -> Rerank")
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Retrieving")):
                query = example["question"]
                
                es_indices = self.search_es(query, topk=RETRIEVAL_TOPK)
                dpr_indices = self.search_dense(query, topk=RETRIEVAL_TOPK)
              
                candidate_indices = list(set(es_indices) | set(dpr_indices))
                
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
    retriever = HybridRetrieval() 
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    retriever.retrieve(query, topk=5)