#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chat.py — Streamlit chat with REAL RAG + LLM
- Tabular retrieval → ranked candidates (best/worst/avg/price/location)
- FAISS passage retrieval → review snippets for LLM grounding with [1],[2] citations
- Always responds; never hallucinates business names
"""

import os, re, html, statistics, unicodedata
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import streamlit as st

# Local utils
from utils.rag import (
    Retriever, build_prompt,
    retrieve_review_passages, build_context_block
)
try:
    from utils.llm_openai import stream_text, complete_text
except Exception:
    stream_text = None
    complete_text = None

# ---------------- Config ----------------
DEFAULT_MIN_STARS = float(os.getenv("CHAT_MIN_STARS", 0.0))  # Lower default to include all restaurants
DEFAULT_K = int(os.getenv("CHAT_K", 8))
USE_LLM = bool(complete_text) and bool(os.getenv("OPENAI_API_KEY"))

# ---------------- UI ----------------
st.set_page_config(page_title="RAG Chat — Odessa & Midland", page_icon="💬", layout="wide")
st.markdown("""<style>
.bubble { padding:12px 14px; margin:8px 0; border-radius:12px; line-height:1.45; width: 100%;}
.bubble-user{ background:#1f6feb22; border:1px solid #1f6feb55; text-align: right;}
.bubble-assist{ background:#30363d; border:1px solid #454c54; text-align: left;}
.small-muted{ color:#9aa4af; font-size:12px; margin:6px 0;}
.bubble a { text-decoration:none; }
.chat-container { max-width: 100%; }
</style>""", unsafe_allow_html=True)
st.title("💬 RAG Chat — Odessa & Midland")

# Controls Row
col1, col2 = st.columns([1, 1])

with col1:
    # GPT Toggle
    if USE_LLM:
        enable_gpt = st.toggle("🤖 Enable AI Insights", value=True, help="Get enhanced responses with GPT-4o-mini")
    else:
        st.info("💡 Add OPENAI_API_KEY to your .env file to enable AI insights")
        enable_gpt = False

with col2:
    # Clear Chat Button
    if st.button("🗑️ Clear Chat", help="Clear all chat history"):
        st.session_state.history = []
        st.rerun()

SYSTEM_PROMPT = """You are an intelligent restaurant assistant specialized in Odessa & Midland.
Use ONLY the supplied candidates and/or the numbered review context. If the answer isn't in the context, say so briefly.
- "Best/Top" → highest rated; "Worst" → lowest rated; "Average" → compute average rating/price/review count.
- Interpret $..$$$$ via price column. Filter by city words (Odessa, Midland).
- Do not invent restaurant names. Prefer items with higher review_count.
- Cite passages with [1], [2] etc. when you use the review context.
- Be accurate about the data you see - if a restaurant shows "26 reviews", don't say it has "no review count data".
- If there are very few restaurants of a specific cuisine type, mention this limitation.
- Focus on the actual restaurants provided in the candidates list."""

# ------------- Helpers -------------
WORD_STRIP = re.compile(r"[^a-z0-9\s'-]+")
def _norm(s:str)->str:
    s = unicodedata.normalize("NFKD", s or "").lower().replace("’","'").replace("`","'")
    s = WORD_STRIP.sub(" ", s)
    return re.sub(r"\s+"," ", s).strip()

FOOD_HINTS = {"restaurant","restaurants","eat","food","breakfast","brunch","lunch","dinner",
    "pizza","burger","taco","bbq","wings","coffee","pho","ramen","mexican","italian","chinese","thai","indian",
    "taqueria","pizzeria","rating","reviews","review","stars","price","$","$$","$$$","$$$$",
    "odessa","midland",
    # Major Fast Food Chains
    "domino","dominos","domino's","mcdonald","mcdonalds","mcdonald's","starbucks","panda","panda express",
    "kfc","popeyes","pizza hut","whataburger","little caesar","little caesars","subway","taco bell",
    "burger king","wendy","wendys","wendy's","chick-fil-a","chik-fil-a","chick fil a","chik fil a",
    "jack in the box","chipotle","five guys","sonic","arby","arbys","arby's","carl","carls jr","carl's jr",
    "denny","dennys","denny's","ihop","wingstop","buffalo wild wings",
    # Casual Dining
    "olive garden","red lobster","applebees","chili","chilis","chili's","outback","texas roadhouse","longhorn",
    # Restaurant Operations
    "hours","opening","open","closed","when does","what time","szechuan","house","cafe","diner","grill","kitchen",
    "hour","time","closing","rating","worst","best","cafe","this restaurant"}
def is_yelp_intent(q:str)->bool:
    qn=_norm(q)
    
    # Check for explicit non-food queries first
    non_food_keywords = {
        "weather", "temperature", "forecast", "rain", "sunny", "cloudy",
        "news", "sports", "politics", "stock", "market", "price of",
        "time", "date", "calendar", "schedule", "appointment",
        "directions", "map", "phone number", "contact",
        "who are you", "what are you", "help", "hello", "hi", "hey"
    }
    
    # Check for non-food keywords with word boundaries to avoid false matches
    for keyword in non_food_keywords:
        if keyword in qn:
            # Check if it's a whole word match to avoid false positives
            import re
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, qn):
                return False
    
    # Check for food-related keywords
    return any(h in qn for h in FOOD_HINTS)

BEST_WORDS={"best","top","highest","top rated"}; WORST_WORDS={"worst","lowest","low rated"}
AVG_WORDS={"avg","average","mean"}; FEW_REVIEWS={"few reviews","least reviews","low reviews","newest"}
LIMIT_RE=re.compile(r"\b(top|show|list)\s+(\d{1,2})\b", re.I)
CITY_RE=re.compile(r"\b(odessa|midland)\b", re.I)
STARS_RE=re.compile(r"(\d(?:\.\d)?)\s*\+?\s*stars?", re.I)
PRICE_RE=re.compile(r"\$|\$\$|\$\$\$|\$\$\$\$|price\s*[:=]?\s*(none|n/a)", re.I)

def parse_limit(q, default_k): 
    m=LIMIT_RE.search(q); 
    return max(1,min(20,int(m.group(2)))) if m else default_k
def parse_cities(q): return list({m.group(1).title() for m in CITY_RE.finditer(q)})
def parse_min_stars(q, d): 
    m=STARS_RE.search(q); 
    return float(m.group(1)) if m else d
def parse_prices(q):
    out=[]; 
    for m in PRICE_RE.finditer(q):
        t=m.group(0).lower()
        out.append("None" if "none" in t or "n/a" in t else "$$$$" if "$$$$" in t else "$$$" if "$$$" in t else "$$" if "$$" in t else "$")
    res=[]; seen=set()
    for p in out:
        if p not in seen: res.append(p); seen.add(p)
    return res
def intent_kind(q):
    qn=_norm(q)
    if any(w in qn for w in BEST_WORDS): return "best"
    if any(w in qn for w in WORST_WORDS): return "worst"
    if any(w in qn for w in AVG_WORDS): return "average"
    if any(w in qn for w in FEW_REVIEWS): return "few_reviews"
    return "generic"

# ------------- Retriever -------------
def get_retriever():
    return Retriever()

retriever = get_retriever()

# ------------- Chat history -------------
if "history" not in st.session_state: st.session_state.history=[]
for role, msg in st.session_state.history:
    if role == "user":
        css = "bubble bubble-user"
        icon = "👤"
        st.markdown(f"<div class='{css}'><strong>{icon} You:</strong><br>{msg}</div>", unsafe_allow_html=True)
    else:
        css = "bubble bubble-assist"
        icon = "🤖"
        st.markdown(f"<div class='{css}'><strong>{icon} Assistant:</strong><br>{msg}</div>", unsafe_allow_html=True)

q = st.chat_input("Ask me — food questions! Try 'best pizza in Odessa' '")
if not q: st.stop()
st.session_state.history.append(("user", html.escape(q)))
st.markdown(f"<div class='bubble bubble-user'><strong>👤 You:</strong><br>{html.escape(q)}</div>", unsafe_allow_html=True)

# ------------- Branch on intent -------------
if not is_yelp_intent(q):
    # For non-food questions, provide helpful AI response if enabled
    if USE_LLM and enable_gpt:
        try:
            # Check if it's a system/identity question
            identity_keywords = ['who are you', 'what are you', 'what is this', 'what do you do', 'introduce yourself', 'tell me about yourself']
            is_identity_question = any(keyword in q.lower() for keyword in identity_keywords)
            
            if is_identity_question:
                # Special response for identity questions
                identity_system_prompt = """You are an AI assistant specialized in analyzing restaurant reviews and providing recommendations for Odessa and Midland, Texas. 
                You help users find the best restaurants, analyze ratings, and provide insights about local dining options.
                Be friendly and explain your purpose clearly."""
                
                result, err = complete_text(
                    [{"role":"system","content": identity_system_prompt},
                     {"role":"user","content": f"User asked: '{q}'. Please introduce yourself as the Odessa/Midland restaurant review analyzer."}],
                    model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                    temperature=0.3,
                    max_tokens=300,
                )
            else:
                # General questions
                general_system_prompt = """You are a helpful assistant for the Odessa & Midland area. 
                You can help with general questions about the area, weather, local information, or anything else.
                Be friendly, helpful, and conversational. If you don't know something specific about Odessa/Midland, 
                say so but still try to be helpful."""
                
                result, err = complete_text(
                    [{"role":"system","content": general_system_prompt},
                     {"role":"user","content": f"User asked: '{q}'. Please provide a helpful response."}],
                    model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
                    temperature=0.3,
                    max_tokens=300,
                )
            
            if result:
                txt = f"💡 **AI Response:** {result.strip()}\n\n_Note: I'm specialized in Odessa & Midland restaurant recommendations! Ask me about food, restaurants, ratings, or dining options in the area._"
            else:
                txt = f"Happy to help — ask me Odessa/Midland food questions or anything else.\n\n_AI insights unavailable: {err}_"
        except Exception as e:
            txt = f"Happy to help — ask me Odessa/Midland food questions or anything else.\n\n_AI insights unavailable: {str(e)}_"
    else:
        txt = "Happy to help — ask me Odessa/Midland food questions or anything else."
    
    st.markdown(f"<div class='bubble bubble-assist'>{txt}</div>", unsafe_allow_html=True)
    st.session_state.history.append(("assistant", txt))
    st.stop()

# ---- Parse NL filters
k = parse_limit(q, DEFAULT_K)
min_stars = parse_min_stars(q, DEFAULT_MIN_STARS)
prices = parse_prices(q)
cities = parse_cities(q)
kind = intent_kind(q)

filters: Dict[str, object] = {"min_stars": min_stars}
if cities: filters["city"] = cities
if prices: filters["price"] = prices

def safe_search(query, k, f):
    try: 
        # For brand-specific queries, don't apply star rating filter
        # Normalize query to handle spaces, apostrophes, and case variations
        normalized_query = query.lower().replace("'", "").replace(" ", "")
        brand_queries = [
            'dominos', 'domino', 'pandaexpress', 'panda', 'starbucks', 'mcdonald', 'mcdonalds', 
            'kfc', 'pizzahut', 'subway', 'tacobell', 'burgerking', 'wendy', 'wendys',
            'chickfila', 'chikfila', 'whataburger', 'jackinthebox', 'popeyes', 'littlecaesar',
            'littlecaesars', 'chipotle', 'fiveguys', 'sonic', 'arby', 'arbys', 'carl',
            'carlsjr', 'denny', 'dennys', 'ihop', 'wingstop', 'buffalowild', 'olivegarden',
            'redlobster', 'applebees', 'chili', 'chilis', 'outback', 'texasroadhouse', 'longhorn'
        ]
        
        if any(brand in normalized_query for brand in brand_queries):
            f = {k: v for k, v in f.items() if k != 'min_stars'}
        
        results = retriever.search(query, k=k, filters=f) or []
        return results
    except Exception as e:
        st.warning(f"Retrieval error: {e}"); return []

# progressive relaxation so we always answer with data
hits = safe_search(q, k, filters) or safe_search(q, k, {**filters, "price": None}) \
    or safe_search(q, k, {"min_stars": filters.get("min_stars",0.0)}) \
    or safe_search("", k, {"min_stars": 0.0})

df = pd.DataFrame(hits)

# ranking tweaks (best/worst/few reviews)
if not df.empty:
    # Check if this is a brand-specific query
    normalized_query = q.lower().replace("'", "").replace(" ", "")
    brand_queries = [
        'dominos', 'domino', 'pandaexpress', 'panda', 'starbucks', 'mcdonald', 'mcdonalds', 
        'kfc', 'pizzahut', 'subway', 'tacobell', 'burgerking', 'wendy', 'wendys',
        'chickfila', 'chikfila', 'whataburger', 'jackinthebox', 'popeyes', 'littlecaesar',
        'littlecaesars', 'chipotle', 'fiveguys', 'sonic', 'arby', 'arbys', 'carl',
        'carlsjr', 'denny', 'dennys', 'ihop', 'wingstop', 'buffalowild', 'olivegarden',
        'redlobster', 'applebees', 'chili', 'chilis', 'outback', 'texasroadhouse', 'longhorn'
    ]
    is_brand_query = any(brand in normalized_query for brand in brand_queries)
    
    if kind == "best" and not is_brand_query:
        # For "best", prioritize restaurants with more reviews for reliability
        # Create a reliability score: rating * log(review_count + 1) to balance rating and review count
        df = df.copy()
        df["reliability_score"] = df["rating"] * np.log(df["review_count"] + 1)
        df = df.sort_values(["reliability_score", "review_count"], ascending=[False, False])
    elif kind == "worst": 
        df = df.sort_values(["rating","review_count"], ascending=[True, True])
    elif kind == "few_reviews": 
        df = df.sort_values(["review_count","rating"], ascending=[True, False])

    # bullets for UI
    def bullets(frame: pd.DataFrame, limit: int) -> str:
        if frame is None or frame.empty:
            return "I couldn't load local rows yet."
        
        # Check if this is a cuisine-specific query with limited results
        cuisine_keywords = ['indian', 'chinese', 'mexican', 'italian', 'thai', 'japanese', 'korean', 'vietnamese']
        is_cuisine_query = any(cuisine in q.lower() for cuisine in cuisine_keywords)
        
        out = []
        
        # Add special message for limited cuisine results
        if is_cuisine_query and len(frame) <= 3:
            cuisine_type = next((cuisine for cuisine in cuisine_keywords if cuisine in q.lower()), 'this cuisine')
            out.append(f"<div style='margin: 12px 0; padding: 12px; border: 1px solid #f0a020; border-radius: 8px; background: #2d1b00; color: #f0a020;'>")
            out.append(f"<strong>📝 Note:</strong> There are only {len(frame)} {cuisine_type} restaurants in our Odessa/Midland database. ")
            out.append(f"Here are all available options:</div>")
        
        for i, (_, r) in enumerate(frame.head(limit).iterrows(), 1):
            # Clean data properly
            name = html.escape(str(r.get("name","Unknown")))
            url = html.escape(str(r.get("url","#")) or "#")
            rating = float(r.get("rating",0.0))
            rc = int(r.get("review_count",0))
            
            # Handle price properly - convert nan to N/A
            price_raw = r.get("price")
            if pd.isna(price_raw) or str(price_raw).lower() in ["nan", "none", ""] or str(price_raw).strip() == "":
                price = "N/A"
            else:
                price = str(price_raw)
            
            city = html.escape(str(r.get("city","")))
            addr = html.escape(str(r.get("address","")))
            
            # Handle categories properly - convert nan to empty string
            categories_raw = r.get("categories","")
            if pd.isna(categories_raw) or str(categories_raw).lower() == "nan" or str(categories_raw).strip() == "":
                categories = ""
            else:
                categories = str(categories_raw)
            
            # Clean up categories display
            cat_display = ""
            if categories and len(categories) < 60:
                # Limit categories to first 2-3 items for readability
                cat_list = categories.split(", ")[:2]
                cat_display = f"<br><small style='color: #9aa4af;'>🍽️ {', '.join(cat_list)}</small>"
            
            # Create organized restaurant card
            restaurant_card = f"""
<div style="margin: 12px 0; padding: 12px; border: 1px solid #454c54; border-radius: 8px; background: #21262d;">
    <div style="display: flex; justify-content: space-between; align-items: start;">
        <div style="flex: 1;">
            <h4 style="margin: 0 0 4px 0; color: #58a6ff;">
                <a href="{url}" target="_blank" style="text-decoration: none; color: #58a6ff;">{i}. {name}</a>
            </h4>
            <div style="color: #f0f6fc; font-size: 14px; margin-bottom: 4px;">
                ⭐ <strong>{rating:.1f}</strong> • {price} • {rc} reviews
            </div>
            <div style="color: #8b949e; font-size: 13px;">
                📍 {city} • {addr}
            </div>
            {cat_display}
        </div>
    </div>
</div>"""
            out.append(restaurant_card)
        
        return "".join(out)

answer = bullets(df, k)

# Average summary (on request)
if kind == "average" and not df.empty:
    ratings = [float(x) for x in df["rating"].dropna().tolist()]
    reviews = [int(x) for x in df["review_count"].dropna().tolist()]
    avg_rating = statistics.mean(ratings) if ratings else 0.0
    avg_reviews = statistics.mean(reviews) if reviews else 0.0
    def p2n(p): return 1 if p=="$" else 2 if p=="$$" else 3 if p=="$$$" else 4 if p=="$$$$" else None
    price_vals = [p2n(str(p)) for p in df["price"] if p2n(str(p)) is not None]
    avg_price = statistics.mean(price_vals) if price_vals else None
    price_txt = "N/A" if avg_price is None else {1:"$",2:"$$",3:"$$$",4:"$$$$"}.get(round(avg_price), "N/A")
    answer += f"\n\n_Averages across shown results → rating: ⭐{avg_rating:.2f}, reviews: {avg_reviews:.0f}, price: {price_txt}_"

# ---- Enhanced GPT Integration ----
llm_block = ""
if USE_LLM and enable_gpt:
    # Try to get review passages for context
    passages = retrieve_review_passages(q, k=6)
    
    # Build context from both search results and review passages
    cand_block = build_prompt(q, df.head(k).to_dict(orient="records")) if not df.empty else ""
    
    if passages:
        # Enhanced context with review passages
        ctx_block, numbered = build_context_block(passages)
        user_msg = f"{cand_block}\n\nReview Context (numbered passages):\n{ctx_block}\n\nAnswer the question using the restaurant candidates and review context. Use bracketed citations like [1], [2] when referencing reviews. Prefer items with higher review counts."
    else:
        # Fallback to restaurant candidates only
        user_msg = f"{cand_block}\n\nAnswer the question about these restaurants. Provide helpful insights about ratings, prices, and locations. Be conversational and helpful."
    
    # Enhanced system prompt
    enhanced_system_prompt = f"""{SYSTEM_PROMPT}

Additional guidelines:
- Be conversational and helpful in your responses
- Provide insights about ratings, prices, and locations
- For "best" restaurants: prioritize those with HIGH review counts (50+ reviews) over high ratings with few reviews
- A restaurant with 5.0 stars but only 1-2 reviews is NOT reliable - prefer restaurants with 4.0+ stars and 20+ reviews
- If asked about "worst" restaurants, explain the rating concerns
- Always mention review counts when discussing reliability: "highly rated with X reviews" vs "perfect rating but only X reviews"
- Mention specific details like review counts and price ranges
- Be encouraging about trying new places"""
    
    try:
        result, err = complete_text(
            [{"role":"system","content": enhanced_system_prompt},
             {"role":"user","content": user_msg}],
            model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
            temperature=0.3,  # Slightly higher for more natural responses
            max_tokens=500,   # More tokens for detailed responses
        )
        if result is None:
            llm_block = f"\n\n💡 _Enhanced insights unavailable: {err}_"
        else:
            llm_block = f"\n\n💡 **AI Insights:** {result.strip()}"
    except Exception as e:
        llm_block = f"\n\n💡 _Enhanced insights unavailable: {str(e)}_"

final = answer + llm_block
st.markdown(f"<div class='bubble bubble-assist'>{final}</div>", unsafe_allow_html=True)
st.session_state.history.append(("assistant", final))
