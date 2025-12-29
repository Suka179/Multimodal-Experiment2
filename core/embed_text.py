from __future__ import annotations
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class TextEmbedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        emb = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return emb.astype("float32")
