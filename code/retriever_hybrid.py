# retriever_hybrid.py
import json
import os
import random
import pickle
import time
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi  # Kiwi 추가

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
        bm25_pickle_name="bm25_hybrid_kiwi.bin",
    ):
        self.dataset_path = dataset_path
        self.context_path = context_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. 데이터 로드
        self.contexts, self.dataset = self.load_data()

        # 2. Kiwi 토크나이저 초기화
        print("🔹 Loading Kiwi tokenizer...")
        self.kiwi = Kiwi()

        # 3. BM25 초기화 (Kiwi 기반)
        print("🔹 Initializing BM25 (Kiwi-based)...")
        self.bm25 = self.init_bm25(bm25_pickle_name)

        # 4. Dense Model 초기화
        print(f"🔹 Loading Dense Model: {dense_model_name} ...")
        self.dense_model = self.init_dense_model(dense_model_name)

        # 5. Passage Embedding 미리 생성
        print("🔹 Encoding contexts with Dense Model (This may take a while)...")
        self.corpus_embeddings = self.dense_model.encode(
            self.contexts,
            batch_size=16,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

    # ====================== 데이터 로드 부분 ======================

    def load_data(self):
        """dataset_path에 따라 validation/test 셋을 유연하게 로드"""
        print("📥 Loading Data...")
        with open(self.context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)
        # 중복 제거
        contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

        org_dataset = load_from_disk(self.dataset_path)

        # 1. test가 있으면 test 사용 (최종 제출용)
        if "test" in org_dataset:
            ds = org_dataset["test"].flatten_indices()
            print("✅ Using 'test' split for prediction.")
        # 2. train + validation 둘 다 있으면 concat (로컬 평가용)
        elif "train" in org_dataset and "validation" in org_dataset:
            print("✅ Using combined 'train' + 'validation' for evaluation.")
            ds = concatenate_datasets(
                [
                    org_dataset["train"].flatten_indices(),
                    org_dataset["validation"].flatten_indices(),
                ]
            )
        # 3. validation만 있으면 validation 사용
        elif "validation" in org_dataset:
            ds = org_dataset["validation"].flatten_indices()
            print("✅ Using 'validation' split only.")
        # 4. 그 외: 첫 번째 split 사용
        elif org_dataset.keys():
            key = list(org_dataset.keys())[0]
            ds = org_dataset[key].flatten_indices()
            print(f"✅ Using single split '{key}' for prediction/evaluation.")
        else:
            raise ValueError("Loaded dataset contains no valid splits.")

        return contexts, ds

    # ====================== Kiwi 토크나이저 ======================

    def kiwi_tokenize(self, text: str):
        """
        Kiwi로 형태소 분석 후, BM25용 토큰 리스트 반환
        """
        if not isinstance(text, str):
            return []
        # token.form 만 사용 (표면형 기준)
        return [token.form for token in self.kiwi.tokenize(text)]

    # ====================== BM25 초기화 ======================

    def init_bm25(self, pickle_name="bm25_hybrid_kiwi.bin"):
        """
        Kiwi 기반 BM25 인덱스 생성 또는 로드
        """
        if os.path.isfile(pickle_name):
            print(f"📂 Loading BM25 index from '{pickle_name}' ...")
            with open(pickle_name, "rb") as f:
                return pickle.load(f)
        else:
            print("🧱 Building BM25 index with Kiwi tokens...")
            tokenized_contexts = [
                self.kiwi_tokenize(doc) for doc in tqdm(self.contexts, desc="Tokenizing contexts (Kiwi)")
            ]
            bm25 = BM25Okapi(tokenized_contexts)
            with open(pickle_name, "wb") as f:
                pickle.dump(bm25, f)
            return bm25

    # ====================== Dense Model 초기화 ======================

    def init_dense_model(self, model_name):
        model = SentenceTransformer(model_name, trust_remote_code=True)
        model.max_seq_length = 512
        model.to(self.device)
        return model

    # ====================== 점수 정규화 ======================

    def min_max_normalize(self, scores):
        scores = np.array(scores, dtype=np.float32)
        if scores.size == 0:
            return scores
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-8:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min + 1e-8)

    # ====================== Hybrid Retrieval ======================

    def retrieve_hybrid(self, top_k=20, alpha=0.4, is_eval=True):
        """
        Hybrid Retrieval + Recall 계산 + Reader용 DataFrame 반환

        - BM25: Kiwi 토크나이저 사용
        - Dense: SentenceTransformer
        - Hybrid: dense_norm * alpha + bm25_norm * (1 - alpha)
        - recall@top_k: 정답 context가 top_k 안에 포함되는 비율
        """
        print(f"\n🚀 Start Hybrid Retrieval (alpha={alpha}, top_k={top_k})")
        queries = self.dataset["question"]
        has_context = "context" in self.dataset.column_names
        has_answers = "answers" in self.dataset.column_names

        # 1. Dense Retrieval (Top-100 후보)
        print("🔍 1) Dense Retrieval (Top-100 candidates)...")
        query_embeddings = self.dense_model.encode(
            queries,
            batch_size=16,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        dense_hits = util.semantic_search(
            query_embeddings, self.corpus_embeddings, top_k=100
        )

        # 2. BM25 Retrieval (Top-100 후보)
        print("🔍 2) BM25 Retrieval (Top-100 candidates, Kiwi-based)...")
        bm25_scores_list = []
        bm25_indices_list = []
        for q in tqdm(queries, desc="BM25 Searching (Kiwi)"):
            tokens = self.kiwi_tokenize(q)
            scores = self.bm25.get_scores(tokens)
            sorted_idx = np.argsort(scores)[::-1][:100]
            bm25_scores_list.append(scores[sorted_idx])
            bm25_indices_list.append(sorted_idx)

        # 3. Hybrid Scoring + Recall + Data 생성
        print("🧮 3) Combining scores & computing Recall...")
        total_retrieved_data = []
        correct_count = 0

        for i in tqdm(range(len(queries)), desc="Hybrid Scoring"):
            doc_score_map = {}

            # --- Dense 부분: 점수 정규화 후 반영 ---
            d_hits = dense_hits[i]
            dense_scores = [h["score"] for h in d_hits]
            dense_norm = self.min_max_normalize(dense_scores)
            for j, h in enumerate(d_hits):
                doc_id = h["corpus_id"]
                score = float(dense_norm[j]) * alpha
                doc_score_map[doc_id] = doc_score_map.get(doc_id, 0.0) + score

            # --- BM25 부분: 점수 정규화 후 반영 ---
            b_scores = bm25_scores_list[i]
            b_indices = bm25_indices_list[i]
            b_norm = self.min_max_normalize(b_scores)
            for j, doc_id in enumerate(b_indices):
                score = float(b_norm[j]) * (1.0 - alpha)
                doc_score_map[doc_id] = doc_score_map.get(doc_id, 0.0) + score

            # --- 최종 Top-k 문서 선택 ---
            sorted_docs = sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
            retrieved_ids = [doc_id for doc_id, _ in sorted_docs]
            retrieved_contexts = [self.contexts[idx] for idx in retrieved_ids]

            # --- Recall 계산 ---
            if is_eval and has_context and has_answers:
                gt_context = self.dataset[i]["context"]
                if gt_context in retrieved_contexts:
                    correct_count += 1

            # --- Reader에게 넘길 데이터 포맷: 문서별로 row 생성 ---
            qid = self.dataset[i]["id"]
            for rank, doc_id in enumerate(retrieved_ids):
                entry = {
                    "id": f"{qid}_{rank}",
                    "question": queries[i],
                    "context": self.contexts[doc_id],
                }
                if has_answers:
                    entry["answers"] = self.dataset[i]["answers"]
                total_retrieved_data.append(entry)

        # --- Recall 출력 ---
        if is_eval and has_context and has_answers:
            recall = correct_count / len(queries)
            print(f"\n✅ Hybrid Retrieval Recall@{top_k} (Kiwi + Dense): {recall:.4f}")

        return pd.DataFrame(total_retrieved_data)


if __name__ == "__main__":
    retriever = HybridRetrieval(
        dataset_path="../data/train_dataset",
        context_path="../data/wikipedia_documents.json",
        dense_model_name="BAAI/bge-m3",
    )
    df = retriever.retrieve_hybrid(top_k=20, alpha=0.4, is_eval=True)
    print("\n✅ Hybrid Retrieval done.")
    print("Returned shape for Reader:", df.shape)
