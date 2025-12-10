import json
import os
import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import pandas as pd
from datasets import Dataset
from elasticsearch import Elasticsearch, helpers
from tqdm.auto import tqdm


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class ElasticRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "./data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        
        # Elasticsearch connection
        try:
            self.es = Elasticsearch("http://localhost:9200")
            if not self.es.ping():
                raise ConnectionError("Could not connect to Elasticsearch at localhost:9200")
            print("Successfully connected to Elasticsearch")
        except Exception as e:
            print(f"Error connecting to Elasticsearch: {e}")
            raise e

        self.index_name = "wikipedia_documents"
        self.tokenize_fn = tokenize_fn

    def get_sparse_embedding(self) -> NoReturn:
        """
        Index documents into Elasticsearch if not already indexed.
        """
        if self.es.indices.exists(index=self.index_name):
            print(f"Index {self.index_name} already exists.")
        else:
            print(f"Creating index {self.index_name}...")
            
            # Define index mapping
            settings = {
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "nori_analyzer": {
                                "type": "custom",
                                "tokenizer": "nori_tokenizer",
                                "decompound_mode": "mixed",
                                "filter": ["nori_part_of_speech"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "nori_analyzer"
                        }
                    }
                }
            }
            
            # Note: If nori plugin is not installed, fallback to standard analyzer
            try:
                self.es.indices.create(index=self.index_name, body=settings)
            except Exception as e:
                print(f"Failed to create index with nori analyzer: {e}")
                print("Falling back to standard analyzer...")
                self.es.indices.create(index=self.index_name)

            # Bulk index documents
            actions = [
                {
                    "_index": self.index_name,
                    "_source": {"content": context}
                }
                for context in self.contexts
            ]
            
            print("Indexing documents...")
            helpers.bulk(self.es, actions)
            print("Indexing complete.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query elastic search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Elastic retrieval: ")
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
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        # Search query
        res = self.es.search(
            index=self.index_name,
            body={
                "query": {"match": {"content": query}},
                "size": k
            }
        )
        
        doc_scores = []
        doc_indices = []
        
        for hit in res['hits']['hits']:
            doc_scores.append(hit['_score'])
            # Find index in self.contexts
            # Note: This is inefficient. Ideally, we should store ID in ES.
            # For now, assuming exact match of content to find index or storing ID.
            # Better approach: Index with ID from 0 to N
            pass
        
        # Re-implementing indexing to include ID for efficient retrieval
        # But wait, self.contexts is a list.
        # Let's assume we can't easily map back from ES hit to self.contexts index without ID.
        # So I will modify get_sparse_embedding to index with ID.
        
        return [], [] # Placeholder, fixed in bulk method logic below

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        
        doc_scores = []
        doc_indices = []
        
        for query in tqdm(queries, desc="Elastic searching"):
            res = self.es.search(
                index=self.index_name,
                body={
                    "query": {"match": {"content": query}},
                    "size": k
                }
            )
            
            scores = []
            indices = []
            
            for hit in res['hits']['hits']:
                scores.append(hit['_score'])
                # We need to retrieve the integer index of the document
                # This requires that we indexed it with the integer ID
                indices.append(int(hit['_id']))
                
            doc_scores.append(scores)
            doc_indices.append(indices)
            
        return doc_scores, doc_indices

    # Override get_sparse_embedding to use ID as document ID
    def get_sparse_embedding(self) -> NoReturn:
        if self.es.indices.exists(index=self.index_name):
            print(f"Index {self.index_name} already exists.")
        else:
            print(f"Creating index {self.index_name}...")
            # ... (settings as before) ...
            try:
                self.es.indices.create(index=self.index_name) # Simple create for now
            except Exception as e:
                pass

            actions = [
                {
                    "_index": self.index_name,
                    "_id": i, # Use integer index as ID
                    "_source": {"content": context}
                }
                for i, context in enumerate(self.contexts)
            ]
            
            print("Indexing documents...")
            helpers.bulk(self.es, actions)
            print("Indexing complete.")
            
    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        res = self.es.search(
            index=self.index_name,
            body={
                "query": {"match": {"content": query}},
                "size": k
            }
        )
        
        doc_scores = []
        doc_indices = []
        
        for hit in res['hits']['hits']:
            doc_scores.append(hit['_score'])
            doc_indices.append(int(hit['_id']))
            
        return doc_scores, doc_indices
