from __future__ import annotations

# ==============================================
# FILE: rag.py (RAG retrieval utilities)
# ==============================================
import os, json, pathlib, re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("faiss-cpu is required. Add it to requirements.txt and reinstall.")

from sentence_transformers import SentenceTransformer

DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "data"))
INDEX_PATH = pathlib.Path(os.getenv("RAG_INDEX", DATA_DIR / "faiss_reviews.index"))
DOCS_PATH = pathlib.Path(os.getenv("RAG_DOCS", DATA_DIR / "reviews_corpus.jsonl"))
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ALLOWED_CITIES = {"odessa", "midland"}

_model: SentenceTransformer | None = None
_index: faiss.IndexFlatIP | None = None
_docstore: List[Dict] | None = None


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _load_docstore() -> List[Dict]:
    global _docstore
    if _docstore is None:
        docs: List[Dict] = []
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        _docstore = docs
    return _docstore


def load_index() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    global _index
    if _index is None:
        if not INDEX_PATH.exists() or not DOCS_PATH.exists():
            raise FileNotFoundError(
                f"Missing RAG files. Run build_rag_index.py. Expected: {INDEX_PATH} and {DOCS_PATH}"
            )
        _index = faiss.read_index(str(INDEX_PATH))
    docs = _load_docstore()
    return _index, docs


def embed(texts: List[str]) -> np.ndarray:
    model = get_model()
    vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    if isinstance(vecs, list):
        vecs = np.array(vecs)
    return vecs.astype("float32")


def _city_ok(meta: Dict, wanted_city: str | None) -> bool:
    if wanted_city:
        return _norm(meta.get("city", "")) == _norm(wanted_city)
    # default: only allow Odessa/Midland
    return _norm(meta.get("city", "")) in ALLOWED_CITIES


def retrieve_context(
    question: str,
    k: int = 12,
    wanted_city: str | None = None,
    business_filter: str | None = None,
) -> List[Dict]:
    """Return top-k review passages with metadata.

    Each item: {"text", "business", "city", "rating", "date", "url", "address"}
    """
    index, docs = load_index()
    qv = embed([question])
    D, I = index.search(qv, k * 5)  # oversample, we will filter by city/brand

    brand = None
    qn = question.lower().replace("’", "'")
    if any(w in qn for w in ["domino's", "dominos", "dominoes", "domino s"]):
        brand = "domino"

    out: List[Dict] = []
    for idx in I[0].tolist():
        if idx < 0:  # faiss pads with -1
            continue
        meta = docs[idx]
        if not _city_ok(meta, wanted_city):
            continue
        if brand and not re.search(r"\bdomino", meta.get("business_norm", "")):
            continue
        out.append(meta)
        if len(out) >= k:
            break
    return out


# Convenience function to turn contexts into a prompt-ready block

def format_context_block(passages: List[Dict]) -> Tuple[str, List[Tuple[int, Dict]]]:
    lines = []
    numbered: List[Tuple[int, Dict]] = []
    for i, p in enumerate(passages, start=1):
        lines.append(
            f"[{i}] {p.get('business','?')} — {p.get('city','')} — ⭐{p.get('biz_rating','?')} — {p.get('date','')}\n"
            f"{p.get('text','').strip()}\nURL: {p.get('url','')}"
        )
        numbered.append((i, p))
    return "\n\n".join(lines), numbered


# ==============================================
# FILE: build_rag_index.py (one-time or refresh)
# ==============================================
from __future__ import annotations
import os, json, pathlib
from typing import List, Dict

import pandas as pd

# Reuse globals from rag.py
# DATA_DIR, INDEX_PATH, DOCS_PATH, MODEL_NAME, embed


def _load_reviews_dataframe() -> pd.DataFrame:
    # Try several common export names from yelp_fetch_reviews.py
    cands = [
        DATA_DIR / "reviews.parquet",
        DATA_DIR / "reviews.csv",
        DATA_DIR / "yelp_reviews.parquet",
        DATA_DIR / "yelp_reviews.csv",
        DATA_DIR / "reviews.jsonl",
        DATA_DIR / "reviews.json",
    ]
    for p in cands:
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            if p.suffix == ".csv":
                return pd.read_csv(p)
            if p.suffix in {".json", ".jsonl"}:
                with open(p, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if "\n{" in text:
                        recs = [json.loads(line) for line in text.splitlines() if line.strip()]
                    else:
                        recs = json.loads(text)
                return pd.DataFrame(recs)
    raise FileNotFoundError("Could not find reviews.* in ./data. Export from yelp_fetch_reviews.py first.")


def _first(*keys):
    def pick(row):
        for k in keys:
            if k in row and pd.notna(row[k]):
                return row[k]
        return None
    return pick


def build():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = _load_reviews_dataframe()

    # Flexible schema mapping
    name_col = next((c for c in ["business_name", "name", "biz_name"] if c in df.columns), None)
    city_col = next((c for c in ["city", "location_city"] if c in df.columns), None)
    url_col = next((c for c in ["url", "business_url", "yelp_url"] if c in df.columns), None)
    addr_col = next((c for c in ["address", "location_address"] if c in df.columns), None)
    biz_rating_col = next((c for c in ["biz_rating", "rating", "stars"] if c in df.columns), None)
    text_col = next((c for c in ["text", "review_text", "review"] if c in df.columns), None)
    date_col = next((c for c in ["date", "time_created", "review_date"] if c in df.columns), None)

    required = [name_col, city_col, text_col]
    if any(x is None for x in required):
        raise ValueError("reviews dataset missing required columns: name/city/text")

    # Keep only Odessa/Midland
    df = df[df[city_col].str.lower().isin({"odessa", "midland"})]

    # Build doc records
    docs: List[Dict] = []
    for _, r in df.iterrows():
        doc = {
            "text": str(r.get(text_col, "")).strip(),
            "business": str(r.get(name_col, "")).strip(),
            "business_norm": str(r.get(name_col, "")).lower(),
            "city": str(r.get(city_col, "")).strip(),
            "url": r.get(url_col, ""),
            "address": r.get(addr_col, ""),
            "biz_rating": float(r.get(biz_rating_col)) if pd.notna(r.get(biz_rating_col)) else None,
            "date": str(r.get(date_col, "")),
        }
        if doc["text"]:
            docs.append(doc)

    # Write corpus jsonl
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Build FAISS index
    passages = [d["text"] for d in docs]
    X = embed(passages)  # normalized float32

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, str(INDEX_PATH))

    print(f"Built RAG index with {len(docs)} passages -> {INDEX_PATH}")


if __name__ == "__main__":
    build()


# ==============================================
# FILE: llm_openai.py (OpenAI wrapper focused on RAG)
# ==============================================
from __future__ import annotations
from typing import List, Dict
import os
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI()

RAG_RULES = (
    "You must answer using ONLY the provided context. If the answer is not in the context, say you don't have enough information. "
    "Cite evidence with bracketed numbers like [1], [2] that map to the sources list. "
    "Prefer items with more reviews when judging quality. Keep answers concise and local to Odessa/Midland."
)


def ask_llm(system: str, user: str, temperature: float = 0.2) -> str:
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return res.choices[0].message.content or ""


def ask_llm_rag(question: str, context_block: str, temperature: float = 0.2) -> str:
    system = RAG_RULES
    user = (
        f"Question: {question}\n\n"
        f"Context (numbered):\n{context_block}\n\n"
        f"Answer with citations like [1], [2]."
    )
    return ask_llm(system, user, temperature)


# ==============================================
# FILE: app.py (Streamlit chat wired to RAG+LLM)
# ==============================================
from __future__ import annotations
import streamlit as st

from chat import answer_query  # uses RAG under the hood
from rag import retrieve_context, format_context_block
from llm_openai import ask_llm_rag

st.set_page_config(page_title="RAG Chat — Odessa & Midland", layout="wide")

st.sidebar.title("Filters for RAG")
use_llm = st.sidebar.toggle("Use LLM (GPT‑4o‑mini)", value=True)
min_stars = st.sidebar.slider("Min stars", 0.0, 5.0, 3.5, 0.5)
k = st.sidebar.slider("Top matches to consider (k)", 3, 20, 8)

st.title("💬 RAG Chat — Odessa & Midland")
st.caption("Ask for things like 'Domino's Odessa', 'worst rated pizza in Odessa', or 'few reviews brunch in Midland'.")

q = st.chat_input("Ask about restaurants… e.g., 'worst rated pizza in Odessa', 'top 3 tacos $$ in Midland'")

if q:
    with st.spinner("Thinking with RAG…"):
        # 1) Deterministic ranked list for UI
        base = answer_query(q, k=k, min_stars=min_stars, use_llm=False)
        st.markdown("**Ranked results** (deterministic):")
        for line in base["items"]:
            st.markdown(f"- {line}")

        # 2) Retrieve review snippets for LLM and show sources
        ctx_passages = retrieve_context(q, k=8)
        ctx_block, numbered = format_context_block(ctx_passages)

        if use_llm:
            llm = ask_llm_rag(q, ctx_block, temperature=0.1)
            st.markdown("\n**LLM answer (RAG-grounded):**")
            st.write(llm)

        # 3) Show sources used for the answer
        with st.expander("Sources (RAG passages)"):
            for i, meta in numbered:
                st.markdown(
                    f"[{i}] **{meta.get('business','?')}** — {meta.get('city','')} — ⭐{meta.get('biz_rating','?')}\n\n"
                    f"_{meta.get('text','').strip()}_\n\n"
                    f"URL: {meta.get('url','')}"
                )
