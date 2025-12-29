from dataclasses import dataclass
from pathlib import Path

@dataclass
class AppConfig:
    # 数据目录（你可以按需改成别的）
    workspace_dir: Path = Path("./data")
    papers_dir: Path = Path("./data/papers")      # 归档后的论文根目录
    images_dir: Path = Path("./data/images")      # 图片库根目录（可选）

    # ChromaDB 持久化目录
    chroma_dir: Path = Path("./data/chroma_db")

    # 集合名
    paper_collection: str = "papers"
    image_collection: str = "images"

    # 模型
    text_embedding_model: str = "./models/all-MiniLM-L6-v2"
    clip_model_name: str = "./models/CLIP"   # ViT-B/32

    # PDF 处理
    max_pages_per_pdf: int = 30           # 控制速度（可调大）
    chunk_chars: int = 1200               # 分段长度
    chunk_overlap: int = 200              # 重叠字符数

    # 检索
    topk_default: int = 5
