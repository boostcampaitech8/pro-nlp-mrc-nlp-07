import json
import logging
import os
import pickle
import time
import random
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi, BM25Plus, BM25L
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

# retrieval.py

class BM25Retriever:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        
        self.tokenize_fn = tokenize_fn  # 토크나이저 함수 저장
        self.data_path = data_path
        
        # 데이터 로드 (SparseRetrieval과 동일)
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )
        logger.info(f"Initialized BM25Retriever with {len(self.contexts)} unique contexts")
        self.ids = list(range(len(self.contexts)))

        self.context_to_id = {
            text: i for i, text in enumerate(self.contexts)
        }

        # ⭐ BM25 전용: Context를 토큰화합니다.
        print("Tokenizing contexts for BM25...")
        self.tokenized_contexts = [
            self.tokenize_fn(doc) for doc in tqdm(self.contexts, desc="Tokenizing")
        ]
        
        self.bm25_model = None
        self.get_bm25_model() # 모델 생성/로드 함수 호출

    def get_bm25_model(self) -> NoReturn:
        """BM25 모델을 만들고 pickle로 저장하거나 로드합니다."""

        pickle_name = f"bm25_model.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            logger.info("Loading pre-computed BM25 model from cache")
            with open(emd_path, "rb") as file:
                self.bm25_model = pickle.load(file)
            print("BM25 model pickle load.")
        else:
            logger.info("Computing BM25 model for passages")
            print("Building BM25 model")
            
            self.bm25_model = BM25Okapi(self.tokenized_contexts)
            
            with open(emd_path, "wb") as file:
                pickle.dump(self.bm25_model, file)
            print("BM25 model pickle saved.")

    # SparseRetrieval의 retrieve 함수 로직을 그대로 가져오되, 내부 스코어링 함수만 BM25 버전으로 대체
    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        
        assert (
            self.bm25_model is not None
        ), "get_bm25_model() 메소드를 먼저 수행해줘야합니다. (BM25 모델 로드)"

        # 단일 쿼리 처리
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_bm25(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        # bulk 쿼리 처리
        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query BM25 search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_bm25(
                    query_or_dataset["question"], k=topk
                )

            # ⭐⭐ 로그용: 실패한 ID를 저장할 리스트 정의
            failed_ids = []
            failed_lens = []
            # Hit@K 계산 및 결과 DataFrame 생성
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="BM25 retrieval: ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]

                    original_context_id = self.context_to_id.get(tmp["original_context"])
                    is_hit_at_k = original_context_id in doc_indices[idx]
                    tmp["is_hit_at_k"] = is_hit_at_k
                    
                    # ⭐⭐⭐ 정답 문서 획득 실패 시 로그 기록 로직 시작 ⭐⭐⭐
                    if not is_hit_at_k:
                        # ⭐⭐ 로그용: 실패 ID 리스트에 현재 ID 추가
                        failed_ids.append(tmp['id'])
                        failed_lens.append(len(tmp['original_context']))

                        # Top-K 문맥 인덱스 리스트 (doc_indices[idx])를 사용하여 점수와 함께 출력
                        doc_scores_list = doc_scores[idx] # 이전에 get_relevant_doc_bulk에서 얻은 점수 리스트
                        
                        logger.error("-" * 60)
                        logger.error(f"🚨 RETRIEVAL FAILURE (K={topk}) for ID: {tmp['id']}")
                        logger.error(f"  QUESTION: {tmp['question']}")
                        logger.error(f"  TARGET CONTEXT (GT): {tmp['original_context']}")
                        logger.error(f"  CONTEXT LENGTH (GT): {len(tmp['original_context'])}")
                        logger.error(f"  ANSWER : {tmp['answers']}")
                        logger.error(f"  --- Top {topk} Retrieved Contexts and Scores ---")
                        
                        # K개의 문맥을 순위, 점수와 함께 로그에 상세 기록
                        scores = []
                        contentLens = []
                        for rank in range(topk):
                            context = self.contexts[doc_indices[idx][rank]]
                            score = doc_scores_list[rank] # 해당 순위의 점수 사용
                            
                            # 점수 & 길이와 함께 출력
                            scores.append(score)
                            contentLens.append(len(context))
                            #logger.error(f"  Rank {rank+1} (Score: {score:.4f}, Length: {len(context)}): {context}")
                        
                        logger.error(f"  Scores: {scores}")
                        logger.error(f"  Content Lengths: {contentLens}")
                        logger.error("-" * 60)
                    # ⭐⭐⭐ 로그 기록 로직 종료 ⭐⭐⭐


                total.append(tmp)

            # ⭐⭐ 로그용: 루프 종료 후 실패 ID 목록 출력 ⭐⭐
            if failed_ids:
                logger.error("=" * 60)
                logger.error(f"🚨 Total {len(failed_ids)} Retrieval Failures (K={topk})")
                logger.error(f"   Failed IDs: {failed_ids}")
                logger.error(f"   Failed Lens: {failed_lens}")
                logger.error("=" * 60)

            return pd.DataFrame(total)

    def get_relevant_doc_bm25(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """하나의 쿼리에 대한 BM25 스코어링"""
        
        tokenized_query = self.tokenize_fn(query)
        
        with timer("query BM25 scoring"):
            doc_scores = self.bm25_model.get_scores(tokenized_query)

        sorted_result = np.argsort(doc_scores)[::-1]
        
        doc_score = doc_scores[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        
        return doc_score, doc_indices

    def get_relevant_doc_bulk_bm25(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """다수의 쿼리에 대한 BM25 스코어링"""
        
        doc_scores_list = []
        doc_indices_list = []
        
        with timer("query BM25 bulk scoring"):
            for query in tqdm(queries, desc="BM25 scoring"):
                tokenized_query = self.tokenize_fn(query)
                doc_scores = self.bm25_model.get_scores(tokenized_query)
                
                sorted_result = np.argsort(doc_scores)[::-1]
                
                doc_scores_list.append(doc_scores[sorted_result].tolist()[:k])
                doc_indices_list.append(sorted_result.tolist()[:k])

        return doc_scores_list, doc_indices_list
    

class BM25RetrieverPlus(BM25Retriever):
    """
    BM25Retriever를 상속받아 BM25+ 알고리즘을 사용하는 클래스
    """
    def get_bm25_model(self) -> NoReturn:
        """
        Overriding: BM25Plus 모델 생성 및 저장
        """
        # ⭐ 저장 파일명을 _plus 로 변경하여 충돌 방지
        pickle_name = f"bm25_plus_model.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            logger.info("Loading pre-computed BM25 Plus model from cache")
            with open(emd_path, "rb") as file:
                self.bm25_model = pickle.load(file)
            print("BM25 Plus model loaded.")
        else:
            logger.info("Computing BM25 Plus model")
            print("Building BM25 Plus model")
            
            # ⭐ BM25Plus 사용
            # 파라미터 튜닝 가능: k1=1.5, b=0.75, delta=1.0 (기본값)
            self.bm25_model = BM25Plus(self.tokenized_contexts, k1=1.5, b=0.75, delta=1.0)
            
            with open(emd_path, "wb") as file:
                pickle.dump(self.bm25_model, file)
            print("BM25 Plus model saved.")

class BM25RetrieverL(BM25Retriever):
    """
    BM25Retriever를 상속받아 BM25L 알고리즘을 사용하는 클래스
    """
    def get_bm25_model(self) -> NoReturn:
        """
        Overriding: BM25L 모델 생성 및 저장
        """
        # ⭐ 저장 파일명을 _l 로 변경하여 충돌 방지
        pickle_name = f"bm25_l_model.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        # BM25L의 기본 파라미터 (사용자가 제공한 코드의 기본값 활용)
        k1_default = 1.5
        b_default = 0.75
        delta_default = 1

        # BM25L 클래스 Import (사용자 환경의 BM25L 구현체를 사용)
        try:
            # 여기서는 BM25L이 rank_bm25가 아닌 다른 모듈에서 사용 가능하다고 가정합니다.
            # 만약 import 문제가 발생한다면, BM25L 구현체를 이 파일에 직접 복사-붙여넣기 해야 합니다.
            # 현재 코드에는 import rank_bm25에서 BM25L을 가져오는 구문이 없으므로,
            # 실행 환경에서 BM25L이 자동으로 인식되거나, 별도로 import 되었다고 가정하고 진행합니다.
            pass
        except NameError:
             logger.error("🚨 BM25L class not found. Ensure BM25L implementation is imported or defined.")
             # 실행 불가능 시 예외 처리 필요

        if os.path.isfile(emd_path):
            logger.info("Loading pre-computed BM25 L model from cache")
            with open(emd_path, "rb") as file:
                self.bm25_model = pickle.load(file)
            # 모델 로드 시 파라미터 정보도 출력하여 확인 용이하게 합니다.
            print(f"BM25 L model loaded (k1={self.bm25_model.k1}, b={self.bm25_model.b}, delta={self.bm25_model.delta}).")
        else:
            logger.info("Computing BM25 L model")
            print(f"Building BM25 L model with k1={k1_default}, b={b_default}, delta={delta_default}")
            
            # ⭐ BM25L 모델 인스턴스화
            self.bm25_model = BM25L(
                self.tokenized_contexts, 
                k1=k1_default, 
                b=b_default, 
                delta=delta_default
            )
            
            with open(emd_path, "wb") as file:
                pickle.dump(self.bm25_model, file)
            print("BM25 L model pickle saved.")

class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        logger.info(f"Initialized SparseRetrieval with {len(self.contexts)} unique contexts")
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # ⭐⭐⭐ 변경 1: Context 텍스트 -> ID 매핑 딕셔너리 생성 ⭐⭐⭐
        self.context_to_id = {
            text: i for i, text in enumerate(self.contexts)
        }

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self) -> NoReturn:
        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            logger.info("Loading pre-computed sparse embeddings from cache")
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            logger.info("Computing sparse embeddings for passages")
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            logger.info(f"Sparse embedding shape: {self.p_embedding.shape}")
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    def build_faiss(self, num_clusters=64) -> NoReturn:
        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            logger.info(f"Loading pre-built FAISS indexer from {indexer_path}")
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            logger.info(f"Building FAISS indexer with {num_clusters} clusters")
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            logger.info(f"FAISS indexer saved to {indexer_path}")
            print("Faiss Indexer Saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert (
            self.p_embedding is not None
        ), "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )

            # ⭐⭐ 로그용: 실패한 ID를 저장할 리스트 정의
            failed_ids = []
            failed_lens = []
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]

                    # ⭐⭐⭐ 변경 2: Hit@K 계산 로직 추가 시작 (Exhaustive Search) ⭐⭐⭐
                    original_context_id = self.context_to_id.get(tmp["original_context"])
                    is_hit_at_k = original_context_id in doc_indices[idx]
                    tmp["is_hit_at_k"] = is_hit_at_k # Hit@K 결과를 DataFrame에 추가
                    
                    # ⭐⭐⭐ 정답 문서 획득 실패 시 로그 기록 로직 시작 ⭐⭐⭐
                    if not is_hit_at_k:
                        # ⭐⭐ 로그용: 실패 ID 리스트에 현재 ID 추가
                        failed_ids.append(tmp['id'])
                        failed_lens.append(len(tmp['original_context']))

                        # Top-K 문맥 인덱스 리스트 (doc_indices[idx])를 사용하여 점수와 함께 출력
                        doc_scores_list = doc_scores[idx] # 이전에 get_relevant_doc_bulk에서 얻은 점수 리스트
                        
                        logger.error("-" * 60)
                        logger.error(f"🚨 RETRIEVAL FAILURE (K={topk}) for ID: {tmp['id']}")
                        logger.error(f"  QUESTION: {tmp['question']}")
                        logger.error(f"  TARGET CONTEXT (GT): {tmp['original_context']}")
                        logger.error(f"  CONTEXT LENGTH (GT): {len(tmp['original_context'])}")
                        logger.error(f"  ANSWER : {tmp['answers']}")
                        logger.error(f"  --- Top {topk} Retrieved Contexts and Scores ---")
                        
                        # K개의 문맥을 순위, 점수와 함께 로그에 상세 기록
                        scores = []
                        contentLens = []
                        for rank in range(topk):
                            context = self.contexts[doc_indices[idx][rank]]
                            score = doc_scores_list[rank] # 해당 순위의 점수 사용
                            
                            # 점수 & 길이와 함께 출력
                            scores.append(score)
                            contentLens.append(len(context))
                            #logger.error(f"  Rank {rank+1} (Score: {score:.4f}, Length: {len(context)}): {context}")
                        
                        logger.error(f"  Scores: {scores}")
                        logger.error(f"  Content Lengths: {contentLens}")
                        logger.error("-" * 60)
                    # ⭐⭐⭐ 로그 기록 로직 종료 ⭐⭐⭐

                total.append(tmp)

            # ⭐⭐ 로그용: 루프 종료 후 실패 ID 목록 출력 ⭐⭐
            if failed_ids:
                logger.error("=" * 60)
                logger.error(f"🚨 Total {len(failed_ids)} Retrieval Failures (K={topk})")
                logger.error(f"   Failed IDs: {failed_ids}")
                logger.error(f"   Failed Lens: {failed_lens}")
                logger.error("=" * 60)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    def retrieve_faiss(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query)
