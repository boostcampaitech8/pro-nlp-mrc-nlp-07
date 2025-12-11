import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi

class HybridRetriever:
    def __init__(
        self,
        data_path: str = "../data",
        context_path: str = "wikipedia_documents.json",
        dense_model_name: str = "BAAI/bge-m3",
        bm25_pickle_name: str = "bm25_hybrid_kiwi.bin",
        # ⭐ [변경] 파일명 충돌 방지를 위해 512로 이름 변경
        dense_pickle_name: str = "dense_embedding_bge_m3_1024.bin", 
    ):
        self.data_path = data_path
        self.context_path = os.path.join(data_path, context_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 위키피디아 문서 로드
        with open(self.context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        
        # 2. Kiwi 및 BM25 초기화
        print("🔹 Initializing Kiwi & BM25...")
        self.kiwi = Kiwi()
        self.bm25 = self.init_bm25(bm25_pickle_name)

        # 3. Dense Model (BGE-M3) 초기화
        print(f"🔹 Loading Dense Model: {dense_model_name}...")
        self.dense_model = SentenceTransformer(dense_model_name, trust_remote_code=True)
        
        # ⭐ [변경] 요청하신 대로 512로 설정
        self.dense_model.max_seq_length = 512 
        self.dense_model.to(self.device)

        # 4. 문서 임베딩 로드/생성
        dense_pickle_path = os.path.join(self.data_path, dense_pickle_name)

        if os.path.isfile(dense_pickle_path):
            print(f"📂 Loading dense embeddings from '{dense_pickle_name}'...")
            self.corpus_embeddings = torch.load(dense_pickle_path, map_location=self.device)
        else:
            print("🔹 Encoding contexts (This may take a while)...")
            self.corpus_embeddings = self.dense_model.encode(
                self.contexts,
                batch_size=32, 
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            print(f"💾 Saving dense embeddings to '{dense_pickle_name}'...")
            torch.save(self.corpus_embeddings, dense_pickle_path)
        
        self.corpus_embeddings = self.corpus_embeddings.to(self.device)

    def kiwi_tokenize(self, text: str):
        return [token.form for token in self.kiwi.tokenize(text)]

    def init_bm25(self, pickle_name):
        pickle_path = os.path.join(self.data_path, pickle_name)
        if os.path.isfile(pickle_path):
            with open(pickle_path, "rb") as f:
                return pickle.load(f)
        else:
            print("🧱 Building BM25 index...")
            tokenized_contexts = [
                self.kiwi_tokenize(doc) for doc in tqdm(self.contexts, desc="Tokenizing")
            ]
            bm25 = BM25Okapi(tokenized_contexts)
            with open(pickle_path, "wb") as f:
                pickle.dump(bm25, f)
            return bm25

    def retrieve(self, dataset, topk=20, rrf_k=60):
        queries = dataset["question"]
        total_data = []
        
        # --- 1. Dense Retrieval ---
        print("🔍 [1/3] Dense Retrieval...")
        instruction = "Represent this sentence for searching relevant passages: "
        q_with_inst = [instruction + q for q in queries]
        
        q_embeddings = self.dense_model.encode(
            q_with_inst, 
            batch_size=32, 
            show_progress_bar=True, 
            convert_to_tensor=True, 
            normalize_embeddings=True
        )
        
        # Dense Top-100
        dense_hits = util.semantic_search(q_embeddings, self.corpus_embeddings, top_k=100)

        # --- 2. BM25 Retrieval ---
        print("🔍 [2/3] BM25 Retrieval...")
        bm25_results = []
        for q in tqdm(queries, desc="BM25"):
            tokens = self.kiwi_tokenize(q)
            scores = self.bm25.get_scores(tokens)
            # 최적화: argsort 대신 argpartition 사용
            top_n_idx = np.argpartition(scores, -100)[-100:]
            top_n_idx = top_n_idx[np.argsort(scores[top_n_idx])[::-1]]
            bm25_results.append(top_n_idx)

        # --- 3. RRF Merge ---
        print("🧮 [3/3] RRF Merging & Result Generation...")
        
        for i in tqdm(range(len(queries)), desc="RRF Processing"):
            doc_score_map = {}

            # (1) Dense Rank Score
            for rank, hit in enumerate(dense_hits[i]):
                doc_id = hit["corpus_id"]
                doc_score_map[doc_id] = doc_score_map.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

            # (2) BM25 Rank Score
            for rank, doc_id in enumerate(bm25_results[i]):
                doc_score_map[doc_id] = doc_score_map.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

            # (3) Sort & Get Top-K
            sorted_docs = sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)[:topk]
            retrieved_ids = [doc_id for doc_id, _ in sorted_docs]
            retrieved_contexts = [self.contexts[idx] for idx in retrieved_ids]

            # Hit@K 계산 (로깅용)
            is_hit = False
            if "context" in dataset.features:
                gt = dataset[i]["context"]
                if gt in retrieved_contexts:
                    is_hit = True

            # ⭐ [유지] Context들을 하나의 문자열로 결합 (MRC 입력 형식 준수)
            joined_context = " ".join(retrieved_contexts)

            base_entry = {
                "id": dataset[i]["id"],
                "question": queries[i],
                "context": joined_context, 
                "is_hit_at_k": is_hit       
            }
            
            if "answers" in dataset.features:
                base_entry["answers"] = dataset[i]["answers"]

            total_data.append(base_entry)

        df = pd.DataFrame(total_data)
        return df