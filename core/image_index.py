from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from PIL import Image

from core.config import AppConfig
from core.utils import ensure_dir, sha1_of_file, list_files
from core.embed_clip import CLIPEmbedder

class ImageService:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        ensure_dir(cfg.workspace_dir)
        ensure_dir(cfg.chroma_dir)

        self.client = chromadb.PersistentClient(
            path=str(cfg.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.col = self.client.get_or_create_collection(
            name=cfg.image_collection,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = CLIPEmbedder(cfg.clip_model_name)

    def add_image(self, img_path: Path) -> Dict[str, Any]:
        img_path = img_path.resolve()
        if not img_path.exists():
            raise FileNotFoundError(str(img_path))

        file_hash = sha1_of_file(img_path)
        existing = self.col.get(where={"file_hash": file_hash}, include=["metadatas"])
        if existing and existing.get("ids"):
            return {"status": "skipped", "reason": "already indexed", "file": str(img_path)}

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            return {"status": "failed", "reason": f"open image error: {e}", "file": str(img_path)}

        emb = self.embedder.encode_images([img], normalize=True)[0].tolist()
        _id = file_hash
        meta = {
            "type": "image",
            "file_hash": file_hash,
            "file_path": str(img_path),
            "file_name": img_path.name,
        }
        self.col.add(ids=[_id], embeddings=[emb], metadatas=[meta], documents=[""])
        return {"status": "ok", "file": str(img_path)}

    def batch_index_images(self, folder: Path) -> Dict[str, Any]:
        folder = folder.resolve()
        imgs = list_files(folder, [".jpg", ".jpeg", ".png", ".webp", ".bmp"])
        results = {"total": len(imgs), "ok": 0, "skipped": 0, "failed": 0, "details": []}
        for p in imgs:
            try:
                r = self.add_image(p)
                if r["status"] == "ok":
                    results["ok"] += 1
                elif r["status"] == "skipped":
                    results["skipped"] += 1
                else:
                    results["failed"] += 1
                results["details"].append(r)
            except Exception as e:
                results["failed"] += 1
                results["details"].append({"status": "failed", "reason": str(e), "file": str(p)})
        return results

    def search_image(self, query: str, topk: int = 5) -> Dict[str, Any]:
        q_emb = self.embedder.encode_text([query], normalize=True)[0].tolist()
        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=topk,
            include=["metadatas", "distances"]
        )

        hits = []
        for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
            sim = 1.0 - float(dist)
            hits.append({
                "file": meta["file_path"],
                "score": round(sim, 4),
            })
        return {"query": query, "results": hits}
