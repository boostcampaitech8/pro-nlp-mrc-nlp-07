import json
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm

# BM25 라이브러리 임포트
from rank_bm25 import BM25Okapi

seed = 2024
random.seed(seed)
np.random.seed(seed)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data",  # [수정] 기본값 ../data로 변경
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
            data_path:
                데이터가 보관되어 있는 경로입니다.
            context_path:
                Passage들이 묶여있는 파일명입니다.
        """
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로 list로 유지
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # 토크나이저 저장 및 BM25 객체 초기화
        self.tokenize_fn = tokenize_fn
        self.bm25 = None

    def get_sparse_embedding(self) -> NoReturn:
        """
        Summary:
            BM25 인덱스를 생성하거나 불러옵니다.
        """
        # 저장할 파일 이름 설정
        pickle_name = "bm25_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("BM25 embedding pickle loaded.")
        else:
            print("Build BM25 embedding...")
            # BM25는 문서를 미리 토큰화해야 함
            tokenized_contexts = [self.tokenize_fn(doc) for doc in tqdm(self.contexts, desc="Tokenizing contexts")]
            self.bm25 = BM25Okapi(tokenized_contexts)
            
            with open(emd_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("BM25 embedding pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.
        """
        # BM25가 로드되지 않았으면 로드
        if self.bm25 is None:
            self.get_sparse_embedding()

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:.4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 context를 합쳐서 반환
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Summary:
            단일 Query에 대해 BM25 검색을 수행합니다.
        """
        tokenized_query = self.tokenize_fn(query)
        
        with timer("query ex search"):
            result = self.bm25.get_scores(tokenized_query)
            
        sorted_result = np.argsort(result)[::-1]
        doc_score = result[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Summary:
            다중 Query에 대해 BM25 검색을 수행합니다.
        """
        doc_scores = []
        doc_indices = []
        
        # BM25는 bulk 처리가 행렬 연산이 아니므로 루프를 돕니다.
        for query in tqdm(queries, desc="BM25 searching"):
            tokenized_query = self.tokenize_fn(query)
            result = self.bm25.get_scores(tokenized_query)
            
            sorted_result = np.argsort(result)[::-1]
            doc_scores.append(result[sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
            
        return doc_scores, doc_indices


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="")
  
    parser.add_argument("--dataset_name", metavar="../data/train_dataset", type=str, default="../data/train_dataset")
    parser.add_argument(
    "--model_name_or_path", 
    metavar="klue/roberta-large",  
    type=str, 
    default="klue/roberta-large")
    parser.add_argument("--data_path", metavar="../data", type=str, default="../data")
    parser.add_argument("--context_path", metavar="wikipedia_documents.json", type=str, default="wikipedia_documents.json")
    
    args = parser.parse_args()

    # 1. 데이터셋 로드
    print("Loading dataset...")
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    # 2. 토크나이저 & Retriever 초기화
    print("Loading tokenizer & retriever...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )
    
    # BM25 인덱싱 생성
    retriever.get_sparse_embedding()

    # 3. 검색 및 Recall 측정
    TOP_K = 20  # 원하는 대로 조절 (10, 20, 50, 100...)
    
    print(f"Start retrieval with Top-{TOP_K}...")
    
    with timer("bulk query search"):
        df = retriever.retrieve(full_ds, topk=TOP_K)
        
        # Recall 계산 로직 (Top-K 문서들 중에 정답이 '포함'되어 있는지 확인)
        df["correct"] = df.apply(lambda x: x["original_context"] in x["context"], axis=1)
        
        print(
            f"Recall@{TOP_K} retrieval result: {df['correct'].sum() / len(df):.4f}"
        )