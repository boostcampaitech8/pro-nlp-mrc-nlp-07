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
from tqdm.auto import tqdm

import torch
from sentence_transformers import SentenceTransformer, util, models

logger = logging.getLogger(__name__)

seed = 2024
random.seed(seed)  # python random seed 고정
np.random.seed(seed)  # numpy random seed 고정


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3", # 사용할 SentenceTransformer 모델 이름
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
        batch_size: int = 16, # 임베딩 시 배치 사이즈
    ) -> NoReturn:
        
        self.model_name = model_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 데이터 로드 (SparseRetrieval/BM25Retriever와 동일)
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )
        logger.info(f"Initialized DenseRetriever with {len(self.contexts)} unique contexts")
        self.ids = list(range(len(self.contexts)))

        self.context_to_id = {
            text: i for i, text in enumerate(self.contexts)
        }
        
        # 모델 로드 및 임베딩 준비
        self.model = self._load_model(model_name)
        self.p_embedding = None # Passage Embedding
        self.indexer = None # FAISS Indexer
        
        self.get_dense_embedding() # 임베딩 계산/로드
        self.build_faiss() # FAISS 인덱스 생성/로드

    def _load_model(self, model_name):
        """SentenceTransformer 모델 로드 함수 (첫 번째 코드 블록에서 가져옴)"""
        try:
            model = SentenceTransformer(model_name, trust_remote_code=True)
        except Exception:
            print(f"Warning: {model_name} 로드 중 에러 발생, Transformer 방식으로 우회 시도...")
            word_embedding_model = models.Transformer(model_name, max_seq_length=512)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        model.max_seq_length = 512
        model.to(self.device)
        return model

    def get_dense_embedding(self) -> NoReturn:
        """Passage Embedding을 만들고 pickle로 저장하거나 로드합니다."""

        pickle_name = f"dense_embedding_{self.model_name.replace('/', '_')}.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            logger.info("Loading pre-computed dense embeddings from cache")
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            logger.info("Computing dense embeddings for passages")
            print("Encoding Passages...")
            
            # SentenceTransformer를 이용해 Passage 임베딩
            self.model.eval()
            with torch.no_grad():
                self.p_embedding = self.model.encode(
                    self.contexts, 
                    batch_size=self.batch_size, 
                    show_progress_bar=True, 
                    convert_to_numpy=True # FAISS를 위해 numpy로 변환
                )
                
            logger.info(f"Dense embedding shape: {self.p_embedding.shape}")
            print(self.p_embedding.shape)

            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def build_faiss(self, n_list=100) -> NoReturn:
        """Faiss indexer에 Passage Embedding을 fitting 시키거나 로드합니다."""

        # indexer 파일명에 모델 이름과 n_list를 포함하여 충돌 방지
        indexer_name = f"faiss_dense_{self.model_name.replace('/', '_')}_nlist{n_list}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)

        if os.path.isfile(indexer_path):
            logger.info(f"Loading pre-built FAISS indexer from {indexer_path}")
            self.indexer = faiss.read_index(indexer_path)
            print("Load Saved Faiss Indexer.")

        else:
            logger.info(f"Building FAISS indexer for dense embeddings with n_list={n_list}")
            
            # FAISS는 float32를 사용
            p_emb = self.p_embedding.astype(np.float32)
            emb_dim = p_emb.shape[-1]
            
            # FAISS Index 생성: IndexFlatL2(L2 거리) 또는 IndexFlatIP(내적) 사용
            # Dense Retrieval에서는 코사인 유사도(Cosine Similarity)가 일반적이며,
            # L2 정규화된 벡터의 내적(IP)은 코사인 유사도와 동일하므로 IndexFlatIP를 사용합니다.
            # BGE-M3의 경우, 코사인 유사도 기반이므로 IndexFlatIP가 적절합니다.
            # IndexFlatIP를 사용하기 위해, 미리 Passage Embedding을 L2 정규화합니다.
            faiss.normalize_L2(p_emb)
            
            # IndexFlatIP를 사용
            self.indexer = faiss.IndexFlatIP(emb_dim) 
            self.indexer.add(p_emb)
            
            # Index 저장
            faiss.write_index(self.indexer, indexer_path)
            logger.info(f"FAISS indexer saved to {indexer_path}")
            print("Faiss Indexer Saved.")
            
            # ⭐ IndexIVF 사용 시 (대규모 데이터셋):
            # quantizer = faiss.IndexFlatIP(emb_dim)
            # self.indexer = faiss.IndexIVFFlat(quantizer, emb_dim, n_list, faiss.METRIC_INNER_PRODUCT)
            # self.indexer.train(p_emb)
            # self.indexer.add(p_emb)
            # self.indexer.nprobe = 10 


    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """Dense Retrieval 수행 함수"""

        assert (
            self.indexer is not None
        ), "build_faiss() 메소드를 먼저 수행해줘야합니다. (FAISS Index 로드)"

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_dense(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:.4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            
            with timer("query dense search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_dense(
                    query_or_dataset["question"], k=topk
                )

            # ⭐⭐ 로그용: 실패한 ID를 저장할 리스트 정의
            failed_ids = []
            failed_lens = []
            
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
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

                        doc_scores_list = doc_scores[idx] 
                        
                        logger.error("-" * 60)
                        logger.error(f"🚨 RETRIEVAL FAILURE (K={topk}) for ID: {tmp['id']}")
                        logger.error(f"  QUESTION: {tmp['question']}")
                        logger.error(f"  TARGET CONTEXT (GT): {tmp['original_context']}")
                        logger.error(f"  CONTEXT LENGTH (GT): {len(tmp['original_context'])}")
                        logger.error(f"  ANSWER : {tmp['answers']}")
                        logger.error(f"  --- Top {topk} Retrieved Contexts and Scores ---")
                        
                        scores = []
                        contentLens = []
                        for rank in range(topk):
                            context = self.contexts[doc_indices[idx][rank]]
                            score = doc_scores_list[rank] 
                            
                            scores.append(score)
                            contentLens.append(len(context))
                        
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


    def get_relevant_doc_dense(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """단일 쿼리에 대한 Dense Embedding 및 FAISS 검색"""
        
        # E5 계열 Prefix 추가 (BAAI/bge-m3도 E5 계열 학습 방식을 따름)
        if "bge" in self.model_name.lower() or "e5" in self.model_name.lower():
            query = f"query: {query}"
        
        self.model.eval()
        with torch.no_grad():
            query_embedding = self.model.encode(
                [query], 
                batch_size=1, 
                show_progress_bar=False, 
                convert_to_numpy=True
            ).astype(np.float32)

        # L2 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(query_embedding)
        
        with timer("query faiss search"):
            # D: Distance (score, 코사인 유사도 값), I: Index (context id)
            D, I = self.indexer.search(query_embedding, k)
        
        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_dense(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """다수 쿼리에 대한 Dense Embedding 및 FAISS 검색"""
        
        # E5 계열 Prefix 추가
        if "bge" in self.model_name.lower() or "e5" in self.model_name.lower():
            queries = [f"query: {q}" for q in queries]
            
        self.model.eval()
        with torch.no_grad():
            query_embeddings = self.model.encode(
                queries, 
                batch_size=self.batch_size, 
                show_progress_bar=True, 
                convert_to_numpy=True
            ).astype(np.float32)
            
        # L2 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(query_embeddings)

        with timer("query faiss bulk search"):
            # D: Distance (score), I: Index (context id)
            D, I = self.indexer.search(query_embeddings, k)

        return D.tolist(), I.tolist()

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
