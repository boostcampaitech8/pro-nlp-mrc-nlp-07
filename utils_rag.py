import re
from typing import List, Optional, Tuple
import numpy as np
import pickle
import os
import torch
from sentence_transformers import util

class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: Optional[List[str]] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ".", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        final_chunks = []
        
        # 재귀적으로 분할
        self._split_text_recursive(text, self.separators, final_chunks)
        
        # 합치기 (너무 작은 조각들은 합쳐서 chunk_size에 가깝게 만듦)
        return self._merge_splits(final_chunks, self.chunk_size, self.chunk_overlap)

    def _split_text_recursive(self, text: str, separators: List[str], final_chunks: List[str]):
        """Recursively split text using separators."""
        if not separators:
            final_chunks.append(text)
            return

        separator = separators[0]
        new_separators = separators[1:]

        if separator == "":
            # 문자 단위 분할 (최후의 수단)
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                final_chunks.append(text[i:i + self.chunk_size])
            return

        splits = text.split(separator)
        for split in splits:
            if len(split) < self.chunk_size:
                final_chunks.append(split)
            else:
                if new_separators:
                    self._split_text_recursive(split, new_separators, final_chunks)
                else:
                    final_chunks.append(split)

    def _merge_splits(self, splits: List[str], chunk_size: int, overlap: int) -> List[str]:
        """Merge small splits into chunks of appropriate size."""
        docs = []
        current_doc = []
        current_len = 0

        for split in splits:
            if not split.strip():
                continue
                
            split_len = len(split)
            
            if current_len + split_len > chunk_size:
                if current_doc:
                    docs.append(" ".join(current_doc))
                
                # Overlap 처리 (이전 청크의 끝부분을 가져옴 - 단순화된 로직)
                # 여기서는 복잡성을 줄이기 위해 단순히 새로운 청크 시작
                current_doc = [split]
                current_len = split_len
            else:
                current_doc.append(split)
                current_len += split_len
        
        if current_doc:
            docs.append(" ".join(current_doc))
            
        return docs


class TorchVectorStore:
    """
    A simple Vector Store using PyTorch tensors for exact nearest neighbor search.
    This replaces Faiss to avoid NumPy 2.0+ compatibility issues.
    """
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.embeddings = None

    def add_embeddings(self, embeddings: np.ndarray):
        """Add embeddings to the store."""
        # Convert to torch tensor
        if isinstance(embeddings, np.ndarray):
            new_embeddings = torch.tensor(embeddings, dtype=torch.float32)
        elif isinstance(embeddings, torch.Tensor):
            new_embeddings = embeddings.float()
        else:
             new_embeddings = torch.tensor(embeddings, dtype=torch.float32)
             
        # Concatenate if exists
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = torch.cat((self.embeddings, new_embeddings), dim=0)
            
        print(f"Added {len(new_embeddings)} vectors to Torch store. Total: {len(self.embeddings)}")

    def search(self, query_embedding: np.ndarray, topk: int = 5) -> Tuple[List[float], List[int]]:
        """Search for nearest neighbors using Dot Product (Cosine Similarity if normalized)."""
        if self.embeddings is None:
            return [], []

        # Ensure query is tensor on correct device
        if isinstance(query_embedding, np.ndarray):
            query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
        elif isinstance(query_embedding, torch.Tensor):
            query_tensor = query_embedding.float()
        else:
             query_tensor = torch.tensor(query_embedding, dtype=torch.float32)
             
        if query_tensor.ndim == 1:
            query_tensor = query_tensor.unsqueeze(0)

        # Move to device
        store_embeddings = self.embeddings.to(self.device)
        query_tensor = query_tensor.to(self.device)

        # Use sentence_transformers utility for semantic search (CosSim)
        # util.semantic_search returns list of list of dicts {'corpus_id': ..., 'score': ...}
        hits = util.semantic_search(query_tensor, store_embeddings, top_k=topk)[0]
        
        scores = [hit['score'] for hit in hits]
        indices = [hit['corpus_id'] for hit in hits]
        
        return scores, indices

    def save(self, path: str):
        """Save embeddings to disk."""
        if self.embeddings is None:
            print("No embeddings to save.")
            return
        
        # Save as torch tensor
        torch.save(self.embeddings.cpu(), path)
        print(f"Index (Tensor) saved to {path}")

    def load(self, path: str):
        """Load embeddings from disk."""
        if not os.path.exists(path):
            print(f"File {path} does not exist.")
            return
            
        self.embeddings = torch.load(path, map_location="cpu")
        print(f"Index loaded from {path}. Shape: {self.embeddings.shape}")
