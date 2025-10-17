#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import html
import unicodedata
from typing import Dict, List, Tuple, Iterable, Set, Optional

import pandas as pd
import streamlit as st

# ---- Your local utils (unchanged contracts) ----
from utils.rag import Retriever, build_prompt
from utils.llm_openai import stream_text  # used only when LLM toggle is ON


# ============================ Page & Styles ============================
st.set_page_config(page_title="RAG Chat — Odessa & Midland", page_icon="💬", layout="wide")
PAGE_CSS = """
<style>
/* Chat bubbles */
.bubble { padding: 12px 14px; margin: 8px 0; border-radius: 12px; line-height: 1.45; }
.bubble-user   { background: #1f6feb22; border: 1px solid #1f6feb55; }
.bubble-assist { background: #30363d;   border: 1px solid #454c54; }
/* Source chips */
.source-pill { display:inline-block; margin:6px 6px 0 0; padding:6px 10px;
               border-radius: 999px; font-size: 12px; background:#2d333b; border:1px solid #444c56; }
.small-muted { color:#9aa4af; font-size:12px; margin-top:8px; margin-bottom:4px; }
/* Links */
.bubble a { text-decoration: none; }
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)
st.title("💬 RAG Chat — Odessa & Midland")


# ============================ Sidebar ============================
with st.sidebar:
    st.header("Filters for RAG")
    USE_LLM = st.toggle("Use LLM (GPT-4o-mini)", True)
    sel_cities = st.multiselect("City", ["Odessa", "Midland"], [])
    sel_prices = st.multiselect("Price", ["$", "$$", "$$$", "$$$$", "None"], [])
    min_stars = st.slider("Min stars", 0.0, 5.0, 3.5, 0.5)
    k_consider = st.slider("Top matches to consider (k)", 3, 20, 8)

    st.caption(
        "Tip: Try queries like **'worst rated pizza in Odessa'**, "
        "**'top 3 tacos $$ in Midland'**, or a brand like **'Domino’s Odessa'**."
    )

# ============================ Helpers ============================
WORD_STRIP = re.compile(r"[^a-z0-9\s'-]+")
STOPWORDS = {
    "the","a","an","and","of","on","in","at","by","for","to","with","from",
    "restaurant","kitchen","cafe","bar","grill","bbq","food","house","shop",
    "market","express","bistro","diner","bakery","coffee","tea","donut","donuts",
    "donut shop","odessa","midland","tx","near","me","best","top","find","good",
}

REQUIRED_COLS = [
    "name","url","rating","review_count","city","address","categories","price"
]

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has all columns used downstream."""
    df = df.copy()
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = "" if col not in ("rating","review_count") else 0
    # normalize numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
    # safe strings
    for c in ["name","url","city","address","categories","price"]:
        df[c] = df[c].astype(str).fillna("")
    return df

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").lower()
    s = s.replace("’", "'").replace("`", "'")
    s = WORD_STRIP.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _sig_tokens(s: str) -> List[str]:
    return [t for t in _norm(s).split() if len(t) >= 2 and t not in STOPWORDS]

def _bigrams(tokens: List[str]) -> Set[str]:
    return {" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)}

def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    df2 = ensure_columns(df)
    if sel_cities:
        df2 = df2[df2["city"].isin(sel_cities)]
    if sel_prices:
        df2 = df2[df2["price"].isin(sel_prices)]
    return df2[df2["rating"] >= float(min_stars)]

def _format_rows(df: pd.DataFrame, limit: int) -> List[str]:
    rows = []
    for _, r in df.head(limit).iterrows():
        name = str(r.get("name","")).strip() or "Unknown"
        url  = str(r.get("url","")).strip() or "#"
        rating = float(r.get("rating",0.0))
        price  = (r.get("price") or "N/A")
        rc     = int(r.get("review_count",0))
        city   = r.get("city","")
        addr   = r.get("address","")
        rows.append(f"- [{name}]({url}) — ⭐{rating:.1f} • {price} • {rc} reviews • {city} • {addr}")
    return rows

def _sources_chips(df: pd.DataFrame, limit: int) -> str:
    df = ensure_columns(df)
    chips = []
    for _, r in df.head(limit).iterrows():
        rating = float(r.get("rating",0.0))
        price  = (r.get("price") or "N/A")
        name   = str(r.get("name","")).strip() or "Unknown"
        url    = str(r.get("url","")).strip() or "#"
        chips.append(
            f"<span class='source-pill'>⭐ {rating:.1f} • {price} — "
            f"<a href='{html.escape(url)}' target='_blank'>{html.escape(name)}</a></span>"
        )
    return "\n".join(chips)

# ================== Greeting + non-food gates ====================
GREETING_PHRASES = {
    "hello", "hi", "hey", "yo", "sup", "good morning",
    "good afternoon", "good evening", "what's up", "how are you"
}

def is_greeting_only(q: str) -> bool:
    qn = _norm(q)
    tokens = set(qn.split())
    if not (any(phrase in qn for phrase in GREETING_PHRASES) or tokens & GREETING_PHRASES):
        return False
    if any(x in qn for x in ["chick fil", "hibachi", "kfc", "pizza", "burger", "restaurant", "taco", "domino"]):
        return False
    return True

CATEGORY_SYNONYMS: Dict[str, List[str]] = {
    "pizza": ["pizza", "pizzeria", "slice", "pie"],
    "burgers": ["burger", "cheeseburger"],
    "bbq": ["bbq", "barbecue", "smokehouse"],
    "sandwich": ["sandwich", "subs", "submarine", "hoagie", "deli"],
    "chinese": ["chinese", "dumpling", "noodles", "szechuan", "hot pot"],
    "coffee": ["coffee", "espresso", "latte", "cafe"],
    "seafood": ["seafood", "fish", "crab", "shrimp", "oyster"],
    "chicken": ["chicken", "fried chicken", "tenders", "wings"],
    "indian": ["indian", "biryani", "curry", "tandoori", "masala"],
    "mexican": ["mexican", "taqueria", "tacos", "burrito", "al pastor", "barbacoa", "carnitas"],
    "breakfast": ["breakfast", "brunch", "pancake", "waffle"],
    "italian": ["italian", "pasta", "trattoria", "ristorante"],
    "desserts": ["dessert", "ice cream", "gelato", "frozen yogurt"],
    "vegan": ["vegan", "plant based"],
    "thai": ["thai", "pad thai", "tom yum"],
    "korean": ["korean", "bibimbap", "kimchi"],
    "vietnamese": ["vietnamese", "pho", "banh mi"],
    "noodles": ["noodles", "ramen", "udon", "soba"],
    "wings": ["wings", "chicken wings", "buffalo wings"],
    "shawarma": ["shawarma"],
    "kebab": ["kebab"],
    "gyro": ["gyro"],
    "steak": ["steak", "steakhouse", "prime rib"],
    "food trucks": ["food truck", "food trucks"],
}

def find_category_terms(text: str) -> List[str]:
    qn = _norm(text)
    found = []
    for canon, syns in CATEGORY_SYNONYMS.items():
        if any(re.search(rf"\b{re.escape(s)}\b", qn) for s in syns + [canon]):
            found.append(canon)
    return found

FOOD_HINTS = {
    "restaurant","restaurants","eat","food","dine","dining","brunch","breakfast","lunch","dinner",
    "menu","cuisine","dish","dishes","takeout","delivery","open now","open",
    "rating","ratings","reviews","review","stars","address","price","$", "$$", "$$$",
    "reservation","drive thru","drive-thru","bar","cafe","bakery","taqueria","pizzeria"
}
FOOD_HINTS |= set(sum(CATEGORY_SYNONYMS.values(), []))

def looks_like_food_intent(q: str) -> bool:
    qn = _norm(q)
    if not qn:
        return False
    if any(h in qn for h in FOOD_HINTS):
        return True
    if re.search(r"\$\$?|\b\d(\.\d)?\s*stars?\b", qn):
        return True
    if re.search(r"\b(odessa|midland)\b", qn) and re.search(r"\b(best|top|find|near|closest|worst|lowest|low rated)\b", qn):
        return True
    return False

# ========================= Retriever =============================
@st.cache_resource(show_spinner=False)
def get_retriever():
    try:
        r = Retriever()
        # Ensure docs shape
        r.docs = ensure_columns(r.docs)
        return r
    except Exception as e:
        return e  # return the exception so we can show a friendly error

retriever = get_retriever()
if isinstance(retriever, Exception):
    st.error(
        "Failed to initialize retrieval backend.\n\n"
        f"Details: {retriever}\n\n"
        "Check that your processed CSVs exist and that utils.rag.Retriever loads them correctly."
    )
    st.stop()

# ========================= Brands ================================
CHAIN_BRANDS: Dict[str, List[str]] = {
    "dominos": ["domino's", "dominos", "domino s", "domin0s", "dominoes"],
    "kfc": ["kfc", "kentucky fried chicken"],
    "mcdonalds": ["mcdonald's","mcdonalds","mc donald","mc-donalds","macdonalds","mac donalds","mcd"],
    "pizza hut": ["pizza hut","pizzahut","pizza-hut"],
    "chick fil a": ["chick-fil-a","chick fil a","chik fil a","chikfila"],
    "popeyes": ["popeyes","popeye's"],
    "whataburger": ["whataburger"],
    "burger king": ["burger king","burger-king"],
    "taco bell": ["taco bell","tacobell"],
    "little caesars": ["little caesars","little caesar","little ceasar"],
    "marcos pizza": ["marco's pizza","marcos pizza","marco s pizza"],
}

@st.cache_resource(show_spinner=False)
def build_dataset_brands(docs: pd.DataFrame) -> Dict[str, Dict[str, Set[str]]]:
    out: Dict[str, Dict[str, Set[str]]] = {}
    for name in docs["name"].dropna().unique():
        full = _norm(name)
        if not full or len(full) < 5:
            continue
        toks = _sig_tokens(name)
        if not toks:
            continue
        bgs = _bigrams(toks)
        if not bgs and len(toks) < 2:
            continue
        out[full] = {"full": {full, full.replace("'", "")}, "bigrams": bgs}
    return out

DATASET_BRANDS = build_dataset_brands(retriever.docs)

def detect_brand(q: str) -> Optional[str]:
    qn = _norm(q)
    for canon, pats in CHAIN_BRANDS.items():
        if any(_norm(p) in qn for p in pats + [canon]):
            return canon
    # dataset-derived brands (full phrase or any bigram)
    for canon, parts in DATASET_BRANDS.items():
        if any((" " in fv and fv in qn) for fv in parts["full"]):
            return canon
        if any(bg in qn for bg in parts["bigrams"]):
            return canon
    return None

# ===================== Sort & Limit parsing =======================
SORT_PHRASES_HIGH = {"best","highest","top","great","good","high rated","top rated"}
SORT_PHRASES_LOW  = {"worst","lowest","low rated","bad","avoid"}
FEW_REVIEWS_PHRASES = {"few reviews","low reviews","new","newest","least reviews","low review"}
LIMIT_RE = re.compile(r"\b(top|show|list)\s+(\d{1,2})\b")

def parse_sort_and_limit(q: str) -> Tuple[str, bool, int]:
    """
    Returns: (sort_col, ascending, limit)
      sort_col in {"rating","review_count"}
      ascending: True for lowest/least
      limit: default 8 or user-specified (1..20)
    """
    qn = _norm(q)
    limit = 8
    m = LIMIT_RE.search(qn)
    if m:
        try:
            limit = max(1, min(20, int(m.group(2))))
        except Exception:
            pass

    # default: rating desc (best first)
    sort_col, ascending = "rating", False

    if any(p in qn for p in SORT_PHRASES_LOW):
        sort_col, ascending = "rating", True
    elif any(p in qn for p in FEW_REVIEWS_PHRASES):
        sort_col, ascending = "review_count", True
    elif any(p in qn for p in SORT_PHRASES_HIGH):
        sort_col, ascending = "rating", False

    return sort_col, ascending, limit

def sort_limit(df: pd.DataFrame, q: str) -> Tuple[pd.DataFrame, int]:
    df = ensure_columns(df)
    col, asc, limit = parse_sort_and_limit(q)
    if col in df.columns:
        df = df.sort_values([col,"rating"], ascending=[asc, asc]).copy()
    else:
        df = df.sort_values(["rating","review_count"], ascending=[False, False]).copy()
    return df, limit

# ===================== Multi-intent splitter =====================
def split_intents(q: str) -> List[str]:
    q = q.strip()
    if " and " in _norm(q) or "," in q or "&" in q or "\n" in q or "?" in q:
        parts = re.split(r"(?:\s+and\s+|&|,|\n|[?])", q, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p and _norm(p)]
        seen = set(); uniq = []
        for p in parts:
            np = _norm(p)
            if np not in seen:
                uniq.append(p); seen.add(np)
        return uniq[:3]
    return [q]

# ====================== Answer primitives ========================
def answer_brand(one_q: str, brand: str) -> Tuple[str, pd.DataFrame]:
    df = retriever.docs
    pats = CHAIN_BRANDS.get(brand, [brand])
    mask = df["name"].fillna("").apply(lambda x: any(_norm(p) in _norm(x) for p in pats + [brand]))
    df = _apply_filters(df[mask])
    if not df.empty:
        df, limit = sort_limit(df, one_q)
        return f"**{one_q.strip()}**\n\n" + "\n".join(_format_rows(df, limit)), df.head(limit)

    # fallback: show category-similar results if brand not present
    cat_terms = []
    if "kfc" in brand:
        cat_terms = ["chicken"]
    elif any(x in brand for x in ["domino","marcos","pizza"]):
        cat_terms = ["pizza"]
    else:
        cat_terms = find_category_terms(one_q) or ["pizza"]

    df2 = retriever.docs
    cat_mask = df2["categories"].str.contains("|".join([re.escape(c) for c in cat_terms]), case=False, na=False)
    df2 = _apply_filters(df2[cat_mask])
    if df2.empty:
        return f"**{one_q.strip()}**\n\nI couldn’t find **{brand.title()}** in the current dataset for these filters.", pd.DataFrame()
    df2, limit = sort_limit(df2, one_q)
    note = f"_No {brand.title()} found in dataset; showing {', '.join(cat_terms)} instead._\n\n"
    return f"**{one_q.strip()}**\n\n{note}" + "\n".join(_format_rows(df2, limit)), df2.head(limit)

def answer_category(one_q: str, cats: List[str]) -> Tuple[str, pd.DataFrame]:
    dfc = retriever.docs
    cat_regex = "|".join([re.escape(c) for c in cats])
    cat_mask = dfc["categories"].str.contains(cat_regex, case=False, na=False)
    dfc = _apply_filters(dfc[cat_mask])
    if dfc.empty:
        return f"**{one_q.strip()}**\n\nI couldn’t find results that match your filters.", pd.DataFrame()
    dfc, limit = sort_limit(dfc, one_q)
    label = ", ".join(cats).title()
    col, asc, _ = parse_sort_and_limit(one_q)
    header = f"Top {limit} {label}:" if not (col == "review_count" and asc) else f"Least-reviewed {label}:"
    return f"**{one_q.strip()}**\n\n{header}\n\n" + "\n".join(_format_rows(dfc, limit)), dfc.head(limit)

def _safe_retrieval(q: str, k: int, filters: dict) -> List[dict]:
    try:
        hits = retriever.search(q, k=k, filters=filters)
        return hits or []
    except Exception as e:
        st.warning(f"Retrieval error: {e}")
        return []

def answer_retrieval_only(one_q: str) -> Tuple[str, pd.DataFrame]:
    filters = {"min_stars": min_stars}
    if sel_cities: filters["city"] = sel_cities
    if sel_prices: filters["price"] = sel_prices
    hits = _safe_retrieval(one_q, k_consider, filters)
    if not hits:
        return f"**{one_q.strip()}**\n\nI couldn’t find results that match your filters.", pd.DataFrame()
    df = ensure_columns(pd.DataFrame(hits))
    df, limit = sort_limit(df, one_q)
    return f"**{one_q.strip()}**\n\n" + "\n".join(_format_rows(df, limit)), df.head(limit)

def rag_llm_answer(one_q: str) -> Tuple[str, pd.DataFrame]:
    # retrieval for context
    filters = {"min_stars": min_stars}
    if sel_cities: filters["city"] = sel_cities
    if sel_prices: filters["price"] = sel_prices
    hits = _safe_retrieval(one_q, k_consider, filters)
    df_hits = ensure_columns(pd.DataFrame(hits)) if hits else pd.DataFrame()

    def bullets(hs: List[dict], q: str) -> str:
        df = ensure_columns(pd.DataFrame(hs)) if hs else pd.DataFrame()
        if df.empty: return "No matches."
        df, limit = sort_limit(df, q)
        out = []
        for _, r in df.head(limit).iterrows():
            name = html.escape(str(r["name"]))
            cats = html.escape(str(r["categories"]))
            rating = float(r["rating"])
            price = (r["price"] or "N/A")
            out.append(f"- **{name}** (⭐{rating:.1f} • {price}) — {cats}")
        return "\n".join(out)

    # Early exit if model key missing or toggle off
    if not USE_LLM or not os.getenv("OPENAI_API_KEY"):
        return f"**{one_q.strip()}**\n\n{bullets(hits, one_q)}", df_hits

    model_name = os.getenv("OPENAI_MODEL","gpt-4o-mini")
    prompt = f"Filters → city: {sel_cities or ['Odessa','Midland']}, price: {sel_prices or 'any'}, min_stars: {min_stars}\n\n{build_prompt(one_q, hits)}"
    messages = [
        {"role":"system","content":"You are a concise, evidence-based Odessa/Midland restaurant assistant. Use ONLY the provided context; if unsure, say you don’t know."},
        {"role":"user","content":prompt},
    ]

    it, err = stream_text(messages, model=model_name, temperature=0.2, max_tokens=320)
    if it is None:
        # Fallback to retrieval bullets if streaming fails
        fallback = bullets(hits, one_q)
        if err:
            fallback = f"(LLM error: {err})\n\n" + fallback
        return f"**{one_q.strip()}**\n\n{fallback}", df_hits

    out = ""
    try:
        for tok in it:
            out += tok
        if not out.strip():
            out = bullets(hits, one_q)
        return f"**{one_q.strip()}**\n\n{out}", df_hits
    except Exception as e:
        return f"**{one_q.strip()}**\n\n(LLM stream error: {e})\n\n{bullets(hits, one_q)}", df_hits

def answer_one_intent(one_q: str) -> Tuple[str, pd.DataFrame]:
    brand = detect_brand(one_q)
    if brand:
        return answer_brand(one_q, brand)
    cats = find_category_terms(one_q)
    if cats:
        return answer_category(one_q, cats)
    if USE_LLM:
        return rag_llm_answer(one_q)
    return answer_retrieval_only(one_q)

# ====================== Chat history UI ==========================
if "history" not in st.session_state:
    st.session_state.history = []

# Show prior messages
for role, msg in st.session_state.history:
    css = "bubble bubble-user" if role == "user" else "bubble bubble-assist"
    st.markdown(f"<div class='{css}'>{msg}</div>", unsafe_allow_html=True)

# =========================== Input ===============================
placeholder = "Ask about restaurants… e.g., 'worst rated pizza in Odessa', 'top 3 tacos $$ in Midland'"
q = st.chat_input(placeholder)

# If no input yet, show gentle nudge and stop (no errors)
if not q:
    st.stop()

# Store & render user bubble
st.session_state.history.append(("user", html.escape(q)))
st.markdown(f"<div class='bubble bubble-user'>{html.escape(q)}</div>", unsafe_allow_html=True)

# gates
if is_greeting_only(q):
    msg = (
        "👋 Hello! I’m your Odessa & Midland restaurant assistant.\n\n"
        "Try:\n"
        "• *Worst rated pizza in Odessa*\n"
        "• *Few reviews Mexican in Midland (top 5)*\n"
        "• *Top 3 tacos under $$ in Midland*\n"
        "• *Domino’s in Odessa*"
    )
    st.markdown(f"<div class='bubble bubble-assist'>{msg}</div>", unsafe_allow_html=True)
    st.session_state.history.append(("assistant", msg))
    st.stop()

if not looks_like_food_intent(q) and not detect_brand(q):
    msg = (
        "🙂 I’m focused on restaurants in Odessa & Midland.\n\n"
        "Ask things like *'worst rated pizza in Odessa'*, *'few reviews brunch in Midland'*, or *'Domino’s Odessa'*."
    )
    st.markdown(f"<div class='bubble bubble-assist'>{msg}</div>", unsafe_allow_html=True)
    st.session_state.history.append(("assistant", msg))
    st.stop()

# ====================== Multi-intent loop =======================
def sources_block(df: pd.DataFrame, title: str = "Sources") -> str:
    if df is None or df.empty:
        return ""
    _, limit = sort_limit(df, q)  # reuse limit display
    return f"<div class='small-muted'>{title}</div>" + _sources_chips(df, limit)

sub_queries = split_intents(q)
sections: List[str] = []
sources_html_parts: List[str] = []

for sq in sub_queries:
    if not _norm(sq):
        continue
    ans, dfans = answer_one_intent(sq)
    sections.append(ans)
    src_html = sources_block(dfans)
    if src_html:
        sources_html_parts.append(src_html)

final_answer = "\n\n".join(sections).strip()
st.markdown(f"<div class='bubble bubble-assist'>{final_answer}</div>", unsafe_allow_html=True)
for s in sources_html_parts:
    st.markdown(s, unsafe_allow_html=True)
st.session_state.history.append(("assistant", final_answer))
