import json
import os
import pickle
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi

class BM25Retrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        self.data_path = data_path
        self.context_path = context_path
        self.tokenize_fn = tokenize_fn
        self.bm25 = None

        # 1. Context 데이터 로드
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(self.contexts)}")

    def get_bm25(self):
        """."""
        pickle_name = "bm25_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            print("Loading BM25 pickle...")
            with open(emd_path, "rb") as file:
                self.bm25 = pickle.load(file)
        else:
            print("Building BM25 index... (This may take a while)")
            # 토크나이징 진행 (시간 소요됨)
            tokenized_corpus = [self.tokenize_fn(doc) for doc in tqdm(self.contexts, desc="Tokenizing Corpus")]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            # 저장
            with open(emd_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("BM25 pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 10
    ) -> pd.DataFrame:
        
        # BM25 로드 확인
        if self.bm25 is None:
            self.get_bm25()

        if isinstance(query_or_dataset, Dataset):
            # 대량 검색 (Dataset)
            total = []
            for idx, example in enumerate(tqdm(query_or_dataset, desc="BM25 Retrieval")):
                query = example["question"]
                tokenized_query = self.tokenize_fn(query)
                
                # 점수 계산 및 정렬
                doc_scores = self.bm25.get_scores(tokenized_query)
                top_n_indices = np.argsort(doc_scores)[::-1][:topk]
                
                # Context 결합
                best_contexts = [self.contexts[i] for i in top_n_indices]
                
                tmp = {
                    "question": query,
                    "id": example["id"],
                    "context": " ".join(best_contexts)
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)
