from __future__ import annotations
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List

class CLIPEmbedder:
    def __init__(self, model_name: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype("float32")

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], normalize: bool = True) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        if normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype("float32")
