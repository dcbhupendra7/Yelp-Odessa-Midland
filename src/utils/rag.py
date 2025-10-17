#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG utilities for the Odessa & Midland Streamlit app.

This module provides the contracts expected by `pages/chat.py`:
- class `Retriever` with attributes:
    - `.docs` -> a pandas DataFrame of businesses
    - `.search(query: str, k: int, filters: dict) -> List[dict]`
- function `build_prompt(question: str, hits: List[dict]) -> str`

It also includes optional FAISS-based retrieval for review passages if
you have built an index, but falls back gracefully to pure pandas/keyword
ranking so the chat UI never crashes during import.

Design goals:
- **Import-safe**: no SyntaxError from type unions; compatible with Python ≥3.8
- **No cross-import spaghetti**: this file contains ONLY RAG helpers —
  not Streamlit pages, not OpenAI wrappers. Keep those in their own files.
"""
from __future__ import annotations

import os
import re
import json
import pathlib
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# -------------------------- Optional deps --------------------------
_FAISS_AVAILABLE = False
try:  # optional: only used if you built a vector index
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    pass

try:  # optional: only used if you built a vector index
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore

# ---------------------------- Constants ----------------------------
DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "data"))
# Tabular business dataset used by the chat UI. Columns are flexible; we normalize.
DOC_TABLE = pathlib.Path(os.getenv("RAG_DOC_TABLE", DATA_DIR / "businesses.csv"))

# Optional review passage index + store (for deep RAG, not required by chat UI)
INDEX_PATH = pathlib.Path(os.getenv("RAG_INDEX", DATA_DIR / "faiss_reviews.index"))
DOCS_PATH = pathlib.Path(os.getenv("RAG_DOCS", DATA_DIR / "reviews_corpus.jsonl"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

ALLOWED_CITIES = {"odessa", "midland"}

# ---------------------------- Utilities ----------------------------
_WORD_STRIP = re.compile(r"[^a-z0-9\s'-]+")
_STOP = {
    "the","a","an","and","of","on","in","at","by","for","to","with","from",
    "restaurant","kitchen","cafe","bar","grill","bbq","food","house","shop",
    "market","express","bistro","diner","bakery","coffee","tea","donut","donuts",
    "donut shop","odessa","midland","tx","near","me","best","top","find","good",
}

REQUIRED_COLS = [
    "name","url","rating","review_count","city","address","categories","price"
]


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("’", "'").replace("`", "'")
    s = _WORD_STRIP.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = "" if col not in ("rating","review_count") else 0
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
    for c in ["name","url","city","address","categories","price"]:
        df[c] = df[c].astype(str).fillna("")
    return df


# ============================== Retriever ==============================
class Retriever:
    """Lightweight tabular retriever used by the chat page.

    It loads a business table (CSV/Parquet/JSON) and performs a simple
    keyword score + rating/review_count sort, honoring city/price filters.
    """

    def __init__(self, table_path: Optional[pathlib.Path] = None) -> None:
        self.table_path = table_path or DOC_TABLE
        self.docs = self._load_table(self.table_path)

    # --- Loading ---
    def _load_table(self, p: pathlib.Path) -> pd.DataFrame:
        if not p.exists():
            # Try a few common alternatives in the data folder
            candidates = [
                DATA_DIR / "businesses.parquet",
                DATA_DIR / "restaurants.parquet",
                DATA_DIR / "restaurants.csv",
                DATA_DIR / "yelp_businesses.csv",
                DATA_DIR / "yelp_businesses.parquet",
            ]
            for c in candidates:
                if c.exists():
                    p = c
                    break
        if not p.exists():
            # Return an empty df with required columns; the UI will show a friendly error.
            return _ensure_columns(pd.DataFrame(columns=REQUIRED_COLS))

        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix == ".jsonl":
            df = pd.read_json(p, lines=True)
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                df = pd.DataFrame(json.load(f))
        else:
            df = pd.read_csv(p)
        df = _ensure_columns(df)
        # Default scope: Odessa & Midland only (unless user provides filter in UI)
        df["city_norm"] = df["city"].str.lower().str.strip()
        df = df[df["city_norm"].isin(ALLOWED_CITIES)] if not df.empty else df
        return df

    # --- Search ---
    def search(self, query: str, k: int = 8, filters: Optional[Dict] = None) -> List[Dict]:
        filters = filters or {}
        df = self.docs
        if df.empty:
            return []

        # Filter by UI constraints
        min_stars = float(filters.get("min_stars", 0.0))
        df = df[df["rating"] >= min_stars]
        if "city" in filters and filters["city"]:
            allow = {c.lower() for c in filters["city"]}
            df = df[df["city"].str.lower().isin(allow)]
        if "price" in filters and filters["price"]:
            df = df[df["price"].isin(filters["price"])]

        # Keyword score: count matches in name + categories
        qn = _norm(query)
        toks = [t for t in qn.split() if len(t) >= 2 and t not in _STOP]
        if toks:
            pat = re.compile(r"|".join(map(re.escape, toks)))
            name_hits = df["name"].str.lower().str.count(pat)
            cat_hits = df["categories"].str.lower().str.count(pat)
            score = name_hits.add(cat_hits, fill_value=0)
        else:
            # No tokens → neutral score
            score = pd.Series(0, index=df.index)

        # Sort: (keyword score desc, rating desc, review_count desc)
        df = df.assign(_score=score)
        df = df.sort_values(["_score", "rating", "review_count"], ascending=[False, False, False])

        rows = df.head(int(max(1, k)))
        return rows[REQUIRED_COLS].to_dict(orient="records")


# ============================ Prompt builder ============================

def build_prompt(question: str, hits: List[Dict]) -> str:
    """Create a compact prompt section summarizing retrieved items.

    Each bullet: **Name** (⭐x.x • price) — categories
    """
    if not hits:
        return f"Question: {question}\n\nNo matching businesses found in the current filters."

    lines = []
    for h in hits:
        name = str(h.get("name", "?"))
        stars = float(h.get("rating", 0.0))
        price = h.get("price", "N/A") or "N/A"
        cats = str(h.get("categories", "")).strip()
        city = h.get("city", "")
        lines.append(f"- **{name}** (⭐{stars:.1f} • {price}) — {cats} — {city}")

    return (
        f"Question: {question}\n\n"
        f"Candidates (top {len(hits)}):\n" + "\n".join(lines)
    )


# ======================== Optional deep RAG bits ========================
# These are safe to import even if FAISS/model files are missing.

_model: Optional["SentenceTransformer"] = None
_index = None  # type: ignore
_docstore: Optional[List[Dict]] = None


def _get_model() -> Optional["SentenceTransformer"]:
    global _model
    if SentenceTransformer is None:
        return None
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _load_docstore() -> List[Dict]:
    global _docstore
    if _docstore is None:
        docs: List[Dict] = []
        if DOCS_PATH.exists():
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        docs.append(json.loads(line))
        _docstore = docs
    return _docstore or []


def _load_index():
    global _index
    if _index is None and _FAISS_AVAILABLE and INDEX_PATH.exists():
        _index = faiss.read_index(str(INDEX_PATH))  # type: ignore
    return _index


def _embed(texts: List[str]) -> Optional[np.ndarray]:
    model = _get_model()
    if model is None:
        return None
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    if isinstance(vecs, list):
        vecs = np.array(vecs)
    return vecs.astype("float32")


def retrieve_review_passages(question: str, k: int = 12, wanted_city: Optional[str] = None) -> List[Dict]:
    """Optional deep-RAG: retrieve review snippets via FAISS, if available.
    Falls back to empty list if the index/model is not present.
    """
    index = _load_index()
    docs = _load_docstore()
    if index is None or not docs:
        return []
    qv = _embed([question])
    if qv is None:
        return []

    D, I = index.search(qv, k * 5)  # oversample; filter by city/brand below

    brand = None
    qn = _norm(question)
    if any(w in qn for w in ["domino's", "dominos", "dominoes", "domino s"]):
        brand = "domino"

    out: List[Dict] = []
    for idx in I[0].tolist():
        if idx < 0:
            continue
        meta = docs[idx]
        city_ok = _norm(meta.get("city", "")) in ALLOWED_CITIES if not wanted_city else _norm(meta.get("city", "")) == _norm(wanted_city)
        if not city_ok:
            continue
        if brand and not re.search(r"\bdomino", meta.get("business_norm", "")):
            continue
        out.append(meta)
        if len(out) >= k:
            break
    return out


def format_context_block(passages: List[Dict]) -> Tuple[str, List[Tuple[int, Dict]]]:
    lines: List[str] = []
    numbered: List[Tuple[int, Dict]] = []
    for i, p in enumerate(passages, start=1):
        lines.append(
            f"[{i}] {p.get('business','?')} — {p.get('city','')} — ⭐{p.get('biz_rating','?')} — {p.get('date','')}\n"
            f"{str(p.get('text','')).strip()}\nURL: {p.get('url','')}"
        )
        numbered.append((i, p))
    return "\n\n".join(lines), numbered
