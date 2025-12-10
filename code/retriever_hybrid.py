import json
import os
import random
import pickle
import time
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_from_disk, concatenate_datasets, DatasetDict, Features, Sequence, Value
from sentence_transformers import SentenceTransformer, util, models
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

# 시드 고정
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class HybridRetrieval:
    def __init__(
        self,
        dataset_path="../data/train_dataset",
        context_path="../data/wikipedia_documents.json",
        dense_model_name="BAAI/bge-m3",
        bm25_tokenizer_name="klue/roberta-large"
    ):
        self.dataset_path = dataset_path
        self.context_path = context_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 데이터 로드
        self.contexts, self.dataset = self.load_data()
        
        # 2. BM25 초기화
        print(f"Loading BM25 Tokenizer: {bm25_tokenizer_name}...")
        self.bm25_tokenizer = AutoTokenizer.from_pretrained(bm25_tokenizer_name, use_fast=False).tokenize
        self.bm25 = self.init_bm25()

        # 3. Dense Model 초기화
        print(f"Loading Dense Model: {dense_model_name}...")
        self.dense_model = self.init_dense_model(dense_model_name)
        
        # 4. Passage Embedding 미리 생성 
        print("Encoding Contexts with Dense Model (This takes time)...")
        self.corpus_embeddings = self.dense_model.encode(
            self.contexts, 
            batch_size=16, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )

    def load_data(self):
        """dataset_path에 따라 validation/test 셋을 유연하게 로드"""
        print("Loading Data...")
        with open(self.context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        
        org_dataset = load_from_disk(self.dataset_path)
        
        if 'test' in org_dataset:
            ds = org_dataset["test"].flatten_indices() 
        else:
            ds = concatenate_datasets([
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices(),
            ])
            
        return contexts, ds

    def init_bm25(self, pickle_name="bm25_hybrid_embedding.bin"):
        # BM25 인덱스 생성 또는 로드
        if os.path.isfile(pickle_name):
            print("Loading BM25 index from pickle...")
            with open(pickle_name, "rb") as f:
                return pickle.load(f)
        else:
            print("Building BM25 index...")
            tokenized_contexts = [self.bm25_tokenizer(doc) for doc in tqdm(self.contexts, desc="Tokenizing for BM25")]
            bm25 = BM25Okapi(tokenized_contexts)
            with open(pickle_name, "wb") as f:
                pickle.dump(bm25, f)
            return bm25

    def init_dense_model(self, model_name):
        # Dense 모델 로드
        try:
            model = SentenceTransformer(model_name, trust_remote_code=True)
        except:
            word_embedding_model = models.Transformer(model_name, max_seq_length=512)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        model.max_seq_length = 512
        model.to(self.device)
        return model

    def min_max_normalize(self, scores):
        # 정규화 로직
        scores = np.array(scores)
        if scores.min() == scores.max():
            return scores
        return (scores - scores.min()) / (scores.max() - scores.min())

    def retrieve_hybrid(self, top_k=20, alpha=0.4, is_eval=True):
        """
        Recall 계산, CSV 저장 후, Reader에게 넘겨줄 DataFrame을 반환합니다.
        """
        print(f"\n🚀 Start Hybrid Retrieval (Alpha={alpha}, Top-{top_k})...")
        
        queries = self.dataset["question"]
        dense_queries = queries 
        
        total_retrieved_data = []
        
        # 1. Dense 검색 (Top-100 후보 추출)
        print("1. Dense Retrieval...")
        query_embeddings = self.dense_model.encode(dense_queries, batch_size=16, show_progress_bar=True, convert_to_tensor=True)
        dense_hits = util.semantic_search(query_embeddings, self.corpus_embeddings, top_k=100)

        # 2. BM25 검색 (Top-100 후보 추출)
        print("2. BM25 Retrieval...")
        bm25_scores_list = []
        bm25_indices_list = []
        
        for query in tqdm(queries, desc="BM25 Searching"):
            tokenized_query = self.bm25_tokenizer(query)
            scores = self.bm25.get_scores(tokenized_query)
            sorted_idxs = np.argsort(scores)[::-1][:100]
            bm25_scores_list.append(scores[sorted_idxs])
            bm25_indices_list.append(sorted_idxs)

        # 3. 점수 결합 (Hybrid)
        print("3. Combining Scores...")
        correct_count = 0
        
        for i in tqdm(range(len(queries)), desc="Hybrid Scoring"):
            doc_score_map = {}
            
            # B. Dense 점수 반영
            for hit in dense_hits[i]:
                doc_id = hit['corpus_id']
                score = hit['score']
                doc_score_map[doc_id] = doc_score_map.get(doc_id, 0) + (score * alpha)
                
            # C. BM25 점수 반영 (정규화 필수!)
            b_scores = bm25_scores_list[i]
            b_indices = bm25_indices_list[i]
            
            b_scores_norm = self.min_max_normalize(b_scores)
            
            for j, doc_id in enumerate(b_indices):
                score = b_scores_norm[j]
                doc_score_map[doc_id] = doc_score_map.get(doc_id, 0) + (score * (1 - alpha))
            
            # D. 최종 정렬 및 Top-K 추출
            sorted_docs = sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
            retrieved_ids = [doc_id for doc_id, score in sorted_docs]
            retrieved_contexts = [self.contexts[idx] for idx in retrieved_ids] # context 내용

            # E. Recall 확인 (is_eval=True and 정답이 있을 때만 실행)
            has_context = "context" in self.dataset.column_names
            has_answers = "answers" in self.dataset.column_names
            
            if is_eval and has_context and has_answers:
                ground_truth = self.dataset[i]["context"]
                if ground_truth in retrieved_contexts:
                    correct_count += 1
            
            # F. 데이터셋 포맷에 맞게 결과 저장 (Reader에게 넘길 데이터)
            data_entry = {
                "id": self.dataset[i]["id"],
                "question": queries[i],
                "context": " ".join(retrieved_contexts), # Top-K context 합치기
            }
            if has_answers:
                data_entry["answers"] = self.dataset[i]["answers"]
            
            total_retrieved_data.append(data_entry)
            
        
        # G. [추가] Recall 출력 및 파일 저장 (is_eval=True 일 때만)
        if is_eval and has_context and has_answers:
            recall = correct_count / len(queries)
            print(f"✅ Hybrid Retrieval Recall@{top_k}: {recall:.4f}")
            
        # [핵심] DataFrame 반환: inference.py가 이 결과를 받아서 Reader에게 넘깁니다.
        return pd.DataFrame(total_retrieved_data)

if __name__ == "__main__":
    # 0.4로 고정하여 Recall 확인용으로 실행합니다.
    retriever = HybridRetrieval()
    retriever.retrieve_hybrid(top_k=20, alpha=0.4, is_eval=True)
    
    print("\n✅ Hybrid Retrieval setup complete. Now run inference.py.")