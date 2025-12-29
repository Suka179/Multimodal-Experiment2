from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple
import shutil
import chromadb
from chromadb.config import Settings

from core.config import AppConfig
from core.utils import ensure_dir, sha1_of_file, safe_filename, list_files
from core.embed_text import TextEmbedder
from core.pdf_utils import extract_pdf_pages, chunk_text_with_page

class PaperService:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        ensure_dir(cfg.workspace_dir)
        ensure_dir(cfg.papers_dir)
        ensure_dir(cfg.chroma_dir)

        self.client = chromadb.PersistentClient(
            path=str(cfg.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.col = self.client.get_or_create_collection(
            name=cfg.paper_collection,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = TextEmbedder(cfg.text_embedding_model)

    def _topic_best_folder(self, pdf_emb: List[float], topics: List[str]) -> Tuple[str, float]:
        topic_embs = self.embedder.encode(topics, normalize=True)
        # cosine 相似度（已归一化 -> 点积）
        import numpy as np
        sims = topic_embs @ np.array(pdf_emb, dtype="float32")
        best_i = int(sims.argmax())
        return topics[best_i], float(sims[best_i])

    def add_paper(self, pdf_path: Path, topics: List[str] | None = None, move: bool = True) -> Dict[str, Any]:
        pdf_path = pdf_path.resolve()
        if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
            raise FileNotFoundError(f"Not a pdf: {pdf_path}")

        file_hash = sha1_of_file(pdf_path)
        # 防止重复导入
        existing = self.col.get(where={"file_hash": file_hash}, include=["metadatas"])
        if existing and existing.get("ids"):
            return {"status": "skipped", "reason": "already indexed", "file": str(pdf_path)}

        pages = extract_pdf_pages(pdf_path, max_pages=self.cfg.max_pages_per_pdf)
        chunks = chunk_text_with_page(pages, self.cfg.chunk_chars, self.cfg.chunk_overlap)
        if not chunks:
            return {"status": "failed", "reason": "empty text extracted", "file": str(pdf_path)}

        # 论文整体 embedding：用所有 chunk 的平均
        texts = [c["text"] for c in chunks]
        embs = self.embedder.encode(texts, normalize=True)

        import numpy as np
        paper_emb = np.mean(embs, axis=0)
        paper_emb = paper_emb / (np.linalg.norm(paper_emb) + 1e-12)

        # 分类：选最相近 topic 文件夹
        dest_folder = "Unsorted"
        best_sim = None
        if topics:
            dest_folder, best_sim = self._topic_best_folder(paper_emb.tolist(), topics)

        dest_dir = (self.cfg.papers_dir / safe_filename(dest_folder)).resolve()
        ensure_dir(dest_dir)

        # 归档文件：以 hash 前缀避免冲突
        dest_name = safe_filename(pdf_path.stem) + f"__{file_hash[:8]}.pdf"
        dest_path = dest_dir / dest_name

        if move:
            shutil.copy2(str(pdf_path), str(dest_path))
        else:
            dest_path = pdf_path

        # 写入 Chroma：按 chunk 存（返回片段与页码）
        ids = []
        metadatas = []
        documents = []
        embeddings = []

        base_meta = {
            "type": "paper_chunk",
            "file_hash": file_hash,
            "file_path": str(dest_path),
            "file_name": dest_path.name,
            "topic": dest_folder,
        }
        if best_sim is not None:
            base_meta["topic_score"] = best_sim

        for i, c in enumerate(chunks):
            cid = f"{file_hash}_{i}"
            ids.append(cid)
            meta = dict(base_meta)
            meta.update({
                "chunk_id": i,
                "page_start": c["page_start"],
                "page_end": c["page_end"],
            })
            metadatas.append(meta)
            documents.append(c["text"])
            embeddings.append(embs[i].tolist())

        try:
            BATCH = 8
            for i in range(0, len(ids), BATCH):
                self.col.add(
                    ids=ids[i:i + BATCH],
                    documents=documents[i:i + BATCH],
                    metadatas=metadatas[i:i + BATCH],
                    embeddings=embeddings[i:i + BATCH],
                )
        except Exception as e:
            return {
                "status": "failed",
                "reason": f"chroma_add_error: {e}",
                "file": str(pdf_path)
            }

        return {
            "status": "ok",
            "archived_to": str(dest_path),
            "topic": dest_folder,
            "topic_score": best_sim,
            "chunks_indexed": len(chunks),
        }

    def batch_organize(self, folder: Path, topics: List[str]) -> Dict[str, Any]:
        folder = folder.resolve()
        pdfs = list_files(folder, [".pdf"])
        results = {"total": len(pdfs), "ok": 0, "skipped": 0, "failed": 0, "details": []}
        for p in pdfs:
            try:
                r = self.add_paper(p, topics=topics, move=True)
                if r["status"] == "ok":
                    results["ok"] += 1
                elif r["status"] == "skipped":
                    results["skipped"] += 1
                else:
                    results["failed"] += 1
                results["details"].append({"file": str(p), **r})
            except Exception as e:
                results["failed"] += 1
                results["details"].append({"file": str(p), "status": "failed", "reason": str(e)})
        return results

    def search_paper(self, query: str, topk: int = 5, return_snippets: bool = True) -> Dict[str, Any]:
        q_emb = self.embedder.encode([query], normalize=True)[0].tolist()
        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=topk,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            sim = 1.0 - float(dist)
            item = {
                "file": meta["file_path"],
                "topic": meta.get("topic"),
                "pages": f'{meta.get("page_start")}-{meta.get("page_end")}',
                "score": round(sim, 4),
            }
            if return_snippets:
                item["snippet"] = doc[:300].replace("\n", " ")
            hits.append(item)

        # 聚合成文件级列表
        file_rank = {}
        for h in hits:
            file_rank.setdefault(h["file"], {"file": h["file"], "best_score": -1, "topic": h["topic"], "best_pages": h["pages"]})
            if h["score"] > file_rank[h["file"]]["best_score"]:
                file_rank[h["file"]]["best_score"] = h["score"]
                file_rank[h["file"]]["best_pages"] = h["pages"]

        files = sorted(file_rank.values(), key=lambda x: x["best_score"], reverse=True)

        return {"query": query, "top_chunks": hits, "top_files": files}
