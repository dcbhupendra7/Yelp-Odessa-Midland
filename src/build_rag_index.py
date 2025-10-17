#!/usr/bin/env python3
"""
Build FAISS index over businesses_ranked.csv (falls back to businesses_clean.csv).
"""

from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

PROC = Path("data/processed")
INP_RANKED = PROC / "businesses_ranked.csv"
INP_CLEAN  = PROC / "businesses_clean.csv"
OUT_DIR = PROC / "rag"
INDEX_PATH = OUT_DIR / "faiss.index"
DOCS_PATH = OUT_DIR / "docstore.parquet"
META_PATH = OUT_DIR / "meta.json"

def make_doc(row: pd.Series) -> str:
    address = ", ".join([str(row.get("address") or ""), str(row.get("city") or ""), str(row.get("state") or ""), str(row.get("zip_code") or "")])
    price = row.get("price") or "N/A"
    cats = row.get("categories") or ""
    R = row.get("rating")
    v = row.get("review_count")
    return (
        f"Name: {row.get('name')}\n"
        f"Categories: {cats}\n"
        f"Price Tier: {price}\n"
        f"Stars: {R} (based on {v} Yelp reviews)\n"
        f"Location: {address}\n"
        f"URL: {row.get('url')}"
    )

def main():
    INP = INP_RANKED if INP_RANKED.exists() else INP_CLEAN
    if not INP.exists():
        raise SystemExit("Missing processed businesses CSV. Run prepare_business_metrics.py first.")
    df = pd.read_csv(INP)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = df.apply(make_doc, axis=1).tolist()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = model.encode(docs, batch_size=64, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(INDEX_PATH))

    df_out = df.copy()
    df_out["text"] = docs
    df_out.to_parquet(DOCS_PATH, index=False)

    META_PATH.write_text(json.dumps({"model":"all-MiniLM-L6-v2","metric":"cosine","rows":len(df_out)}, indent=2))
    print(f"✓ RAG index built: {len(df_out)} rows -> {INDEX_PATH}")

if __name__ == "__main__":
    main()
