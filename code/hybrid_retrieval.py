# 신지님 psj-01 branch 참고
# https://www.notion.so/Hybrid_Retrieval-code-2c51175ac04f80deb420d6378f2ec2a5?source=copy_link


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
        bm25_tokenizer_name="klue/roberta-large",
        is_eval=False
    ):
        self.dataset_path = dataset_path
        self.context_path = context_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_eval = is_eval
        
        # 1. 데이터 로드
        self.contexts, self.dataset, self.dataset_split = self.load_data(is_eval=is_eval)
        
        # 2. BM25 초기화
        print(f"Loading BM25 Tokenizer: {bm25_tokenizer_name}...")
        self.bm25_tokenizer = AutoTokenizer.from_pretrained(bm25_tokenizer_name, use_fast=False).tokenize
        self.bm25 = self.init_bm25()

        # 3. Dense Model 초기화
        print(f"Loading Dense Model: {dense_model_name}...")
        self.dense_model = self.init_dense_model(dense_model_name)
        self.dense_model_name = dense_model_name
        
        # 4. Passage Embedding 미리 생성 또는 로드
        print("Initializing corpus embeddings...")
        self.corpus_embeddings = self.init_corpus_embeddings()

    # ======================== retriever_hybrid.py 파일 (load_data 함수만) ========================

    def load_data(self, is_eval=False):
        """dataset_path에 따라 validation/test 셋을 유연하게 로드"""
        print("Loading Data...")
        with open(self.context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        
        org_dataset = load_from_disk(self.dataset_path)
        
        # ------------------ [수정된 로직 시작] ------------------
        
        # is_eval=True일 때는 validation split만 사용 (train.py와 동일)
        if is_eval:
            if 'validation' in org_dataset:
                ds = org_dataset["validation"].flatten_indices()
                dataset_split = "validation"
                print("Using 'validation' split for evaluation (do_eval=True).")
            else:
                raise ValueError("--do_eval requires a validation dataset, but 'validation' split not found.")
        
        # 1. 'test' 키가 있으면 test split 사용 (최종 제출 시)
        elif 'test' in org_dataset:
            ds = org_dataset["test"].flatten_indices() 
            dataset_split = "test"
            print("Using 'test' split for prediction.")

        # 4. 그 외의 경우 (예외 처리)
        elif org_dataset.keys():
            key = list(org_dataset.keys())[0]
            dataset_split = key
            print(f"Using single split '{key}' for prediction/evaluation.")
            ds = org_dataset[key].flatten_indices()
        else:
             raise ValueError("Loaded dataset contains no valid splits.")
        
        # ------------------ [수정된 로직 끝] ------------------
            
        return contexts, ds, dataset_split
        
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
    
    def get_embedding_filename(self):
        """임베딩 파일명 생성 (모델명, context 경로, 데이터셋 split 기반)"""
        # 모델명에서 파일명에 사용할 수 없는 문자 제거
        model_safe_name = self.dense_model_name.replace("/", "_").replace("\\", "_")
        # context_path의 파일명 사용
        context_basename = os.path.basename(self.context_path).replace(".json", "")
        # 데이터셋 split 포함 (is_eval일 때만 split 구분)
        if self.is_eval: 
            embedding_filename = f"corpus_embeddings_{model_safe_name}_{context_basename}_{self.dataset_split}.pt"
        else:
            embedding_filename = f"corpus_embeddings_{model_safe_name}_{context_basename}.pt"
        return embedding_filename
    
    def init_corpus_embeddings(self):
        """Passage Embedding 생성 또는 로드"""
        embedding_filename = self.get_embedding_filename()
        print(f"Checking for cached embeddings: {embedding_filename}")
        
        if os.path.isfile(embedding_filename):
            print(f"✅ Found cached embeddings! Loading from {embedding_filename}...")
            try:
                # GPU에 저장된 경우를 대비해 map_location 설정
                corpus_embeddings = torch.load(embedding_filename, map_location=self.device)
                print(f"✅ Successfully loaded {len(corpus_embeddings)} embeddings from cache.")
                return corpus_embeddings
            except Exception as e:
                print(f"⚠️  Failed to load embeddings from {embedding_filename}: {e}")
                print("   Generating new embeddings...")
        else:
            print(f"❌ Embedding file not found: {embedding_filename}")
            print("   Encoding Contexts with Dense Model (This takes time)...")
        
        # 임베딩 생성
        corpus_embeddings = self.dense_model.encode(
            self.contexts, 
            batch_size=16, 
            show_progress_bar=True, 
            convert_to_tensor=True
        )
        
        # 임베딩 저장
        print(f"Saving corpus embeddings to {embedding_filename}...")
        try:
            torch.save(corpus_embeddings, embedding_filename)
            print(f"✅ Successfully saved embeddings to {embedding_filename}")
        except Exception as e:
            print(f"⚠️  Failed to save embeddings: {e}")
        
        return corpus_embeddings

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