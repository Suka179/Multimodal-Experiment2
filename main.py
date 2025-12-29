from __future__ import annotations
import argparse
from pathlib import Path
import json
from core.config import AppConfig
from core.paper_index import PaperService
from core.image_index import ImageService
from core.utils import ensure_dir
import warnings
warnings.filterwarnings(
    "ignore",
    message="Torch was not compiled with flash attention"
)

def parse_topics(s: str | None):
    if not s:
        return None
    return [t.strip() for t in s.split(",") if t.strip()]

def cmd_add_paper(args):
    cfg = AppConfig()
    ensure_dir(cfg.workspace_dir)
    svc = PaperService(cfg)
    topics = parse_topics(args.topics)
    out = svc.add_paper(Path(args.path), topics=topics, move=(not args.no_move))
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

def cmd_search_paper(args):
    cfg = AppConfig()
    svc = PaperService(cfg)
    out = svc.search_paper(args.query, topk=args.topk, return_snippets=(not args.files_only))
    if args.files_only:
        # 只输出文件列表
        files = [x["file"] for x in out["top_files"]]
        print(json.dumps({"query": args.query, "files": files}, ensure_ascii=False, indent=2), flush=True)
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

def cmd_batch_organize(args):
    cfg = AppConfig()
    svc = PaperService(cfg)
    topics = parse_topics(args.topics)
    if not topics:
        raise ValueError("batch_organize requires --topics")
    out = svc.batch_organize(Path(args.folder), topics=topics)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

def cmd_index_images(args):
    cfg = AppConfig()
    svc = ImageService(cfg)
    out = svc.batch_index_images(Path(args.folder))
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

def cmd_search_image(args):
    cfg = AppConfig()
    svc = ImageService(cfg)
    out = svc.search_image(args.query, topk=args.topk)
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)

def build_parser():
    p = argparse.ArgumentParser("Local Multimodal AI Agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    # add_paper
    sp = sub.add_parser("add_paper", help="Add & classify a single pdf paper")
    sp.add_argument("path", type=str, help="Path to pdf")
    sp.add_argument("--topics", type=str, default=None, help='Comma topics, e.g. "CV,NLP,RL"')
    sp.add_argument("--no-move", action="store_true", help="Do not copy/move file into archive, index in-place")
    sp.set_defaults(func=cmd_add_paper)

    # search_paper
    sp = sub.add_parser("search_paper", help="Semantic search papers")
    sp.add_argument("query", type=str)
    sp.add_argument("--topk", type=int, default=5)
    sp.add_argument("--files-only", action="store_true", help="Only return file list (no snippets)")
    sp.set_defaults(func=cmd_search_paper)

    # batch_organize
    sp = sub.add_parser("batch_organize", help="Organize a folder of pdfs into topic subfolders")
    sp.add_argument("folder", type=str)
    sp.add_argument("--topics", type=str, required=True, help='Comma topics, e.g. "CV,NLP,RL"')
    sp.set_defaults(func=cmd_batch_organize)

    # index_images
    sp = sub.add_parser("index_images", help="Index all images in folder into Chroma")
    sp.add_argument("folder", type=str)
    sp.set_defaults(func=cmd_index_images)

    # search_image
    sp = sub.add_parser("search_image", help="Text-to-image search in local image库")
    sp.add_argument("query", type=str)
    sp.add_argument("--topk", type=int, default=5)
    sp.set_defaults(func=cmd_search_image)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
