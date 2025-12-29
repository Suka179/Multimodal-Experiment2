from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Iterable

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def safe_filename(name: str, max_len: int = 180) -> str:
    # 简易清理非法字符
    bad = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for c in bad:
        name = name.replace(c, "_")
    name = name.strip()
    if len(name) > max_len:
        name = name[:max_len]
    return name

def list_files(root: Path, exts: Iterable[str]) -> list[Path]:
    exts = {e.lower() for e in exts}
    out = []
    for dp, _, files in os.walk(root):
        for fn in files:
            p = Path(dp) / fn
            if p.suffix.lower() in exts:
                out.append(p)
    return out
