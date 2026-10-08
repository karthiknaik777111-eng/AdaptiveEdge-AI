"""
Local Dense Embedding Service for Semantic Routing.
Uses sentence-transformers (all-MiniLM-L6-v2) cached locally in models/embeddings.
Runs 100% locally on CPU without external API calls.
"""

import os
import time
import numpy as np
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer


class LocalEmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: str = "models/embeddings"):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self):
        """Initialize the local embedding model on CPU."""
        os.makedirs(self.cache_folder, exist_ok=True)
        # Load local model on CPU
        self.model = SentenceTransformer(
            self.model_name,
            device="cpu",
            cache_folder=self.cache_folder
        )

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text or list of texts into dense vectors.
        Returns normalized float32 numpy array for fast dot-product cosine similarity.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        return embeddings.astype(np.float32)

    def compute_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two normalized vectors."""
        if vec_a.ndim > 1:
            vec_a = vec_a.flatten()
        if vec_b.ndim > 1:
            vec_b = vec_b.flatten()
        dot_product = float(np.dot(vec_a, vec_b))
        return max(0.0, min(1.0, dot_product))


# Global instance
embedding_service: Optional[LocalEmbeddingService] = None

def get_embedding_service() -> LocalEmbeddingService:
    global embedding_service
    if embedding_service is None:
        embedding_service = LocalEmbeddingService()
    return embedding_service
