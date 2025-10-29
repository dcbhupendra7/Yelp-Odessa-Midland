
"""
utils/rag.py
- Loads your businesses table (default: data/processed/rag/businesses_clean.csv)
- Exposes a tabular Retriever() for ranking & filters
- Exposes passage RAG via retrieve_review_passages() using FAISS + SentenceTransformer
- Provides build_prompt() and build_context_block() for LLM grounding
"""

from __future__ import annotations
import os, re, json, pathlib
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import pandas as pd

# ---------- Paths ----------
_THIS = pathlib.Path(__file__).resolve()
ROOT = _THIS.parents[2] if len(_THIS.parents) >= 3 else _THIS.parents[0]
DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", ROOT / "data"))

def _first_existing(paths) -> Optional[pathlib.Path]:
    for p in paths:
        if not p: 
            continue
        pth = pathlib.Path(p).expanduser()
        if pth.exists():
            return pth.resolve()
    return None

DOC_TABLE = _first_existing([
    os.getenv("RAG_DOC_TABLE", "").strip(),
    DATA_DIR / "processed" / "businesses_ranked.csv",
    DATA_DIR / "processed" / "businesses_clean.csv",
    DATA_DIR / "processed" / "rag" / "businesses_clean.csv",
    DATA_DIR / "processed" / "rag" / "businesses_ranked.csv",
    DATA_DIR / "businesses.csv",
])

# FAISS review index & docstore (jsonl/parquet)
INDEX_PATH = _first_existing([
    os.getenv("RAG_INDEX_PATH", "").strip(),
    DATA_DIR / "processed" / "rag" / "faiss.index",
])
DOCSTORE_PATH = _first_existing([
    os.getenv("RAG_DOCSTORE_PATH", "").strip(),
    DATA_DIR / "processed" / "rag" / "docstore.parquet",
    DATA_DIR / "processed" / "rag" / "docstore.jsonl",
])

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

REQUIRED_COLS = ["name","url","rating","review_count","city","address","categories","price","latitude","longitude","id","hours"]
_WORD_STRIP = re.compile(r"[^a-z0-9\s'-]+")

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

def _norm(s: str) -> str:
    s = (s or "").lower().replace("'","'").replace("`","'")
    s = _WORD_STRIP.sub(" ", s)
    return re.sub(r"\s+"," ", s).strip()

def fuzzy_match_score(query: str, target: str, threshold: float = 0.6) -> float:
    """Calculate fuzzy match score between query and target string."""
    query_norm = _norm(query)
    target_norm = _norm(target)
    
    # Direct substring match gets highest score
    if query_norm in target_norm:
        return 1.0
    
    # Use SequenceMatcher for fuzzy matching
    similarity = SequenceMatcher(None, query_norm, target_norm).ratio()
    return similarity if similarity >= threshold else 0.0

def find_fuzzy_matches(query: str, names: List[str], threshold: float = 0.6, max_results: int = 5) -> List[Tuple[str, float]]:
    """Find fuzzy matches for a query against a list of names."""
    matches = []
    for name in names:
        score = fuzzy_match_score(query, name, threshold)
        if score > 0:
            matches.append((name, score))
    
    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:max_results]

def clean_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up missing data by replacing 'nan' strings and empty values with user-friendly text."""
    df = df.copy()
    
    # Define replacement mappings for different fields
    replacements = {
        'price': {
            'nan': 'Call for Info',
            '': 'Call for Info',
            'None': 'Call for Info'
        },
        'address': {
            'nan': 'Address Not Available',
            '': 'Address Not Available'
        },
        'hours': {
            'nan': 'Hours Not Available',
            '': 'Hours Not Available',
            'Hours not available': 'Hours Not Available'
        },
        'categories': {
            'nan': 'Categories Not Available',
            '': 'Categories Not Available'
        },
        'city': {
            'nan': 'City Not Available',
            '': 'City Not Available'
        },
        'name': {
            'nan': 'Restaurant Name Not Available',
            '': 'Restaurant Name Not Available'
        }
    }
    
    # Apply replacements
    for column, replacement_map in replacements.items():
        if column in df.columns:
            # Replace string 'nan' values
            df[column] = df[column].astype(str)
            for old_val, new_val in replacement_map.items():
                df[column] = df[column].replace(old_val, new_val)
    
    return df

# ======================= Tabular Retriever =======================
class Retriever:
    """Simple keyword + rating ranker over the businesses table."""
    def __init__(self, table_path: Optional[pathlib.Path] = None) -> None:
        self.table_path = pathlib.Path(table_path) if table_path else (DOC_TABLE or DATA_DIR / "processed" / "rag" / "businesses_clean.csv")
        self._doc_path = str(self.table_path)
        self._doc_mtime: Optional[float] = None
        self.docs = self._load_table(self.table_path)

    def _load_table(self, p: pathlib.Path) -> pd.DataFrame:
        self._doc_path = str(p)
        try:
            self._doc_mtime = p.stat().st_mtime
        except Exception:
            self._doc_mtime = None
        if not p.exists():
            df = _ensure_columns(pd.DataFrame(columns=REQUIRED_COLS))
            df["__doc_path__"] = self._doc_path
            return df

        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix == ".jsonl":
            df = pd.read_json(p, lines=True)
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f: df = pd.DataFrame(json.load(f))
        else:
            # auto-detect delimiter just in case
            try:
                df = pd.read_csv(p, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(p)

        # map alternates → canonical
        colmap = {
            "business_name":"name","biz_name":"name",
            "stars":"rating","rating_value":"rating",
            "reviewCount":"review_count","reviews":"review_count",
            "price_level":"price","price_tier":"price",
            "business_url":"url","yelp_url":"url","website":"url",
            "location":"address","addr":"address",
            "categories_list":"categories","category":"categories","top_category":"categories",
            "fetched_city":"city",
        }
        for src, dst in colmap.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]

        df = _ensure_columns(df)
        df = clean_missing_data(df)  # Clean up missing data
        df["city_norm"] = df["city"].astype(str).str.lower()
        return df

    def _maybe_reload(self) -> None:
        """Hot-reload the table if file timestamp changed (keeps data fresh)."""
        try:
            p = pathlib.Path(self._doc_path)
            mtime = p.stat().st_mtime
            if self._doc_mtime is None or mtime != self._doc_mtime:
                self.docs = self._load_table(p)
        except Exception:
            # If stat fails for any reason, keep current docs
            pass

    def search(self, query: str, k: int = 8, filters: Optional[Dict] = None) -> List[Dict]:
        self._maybe_reload()
        filters = filters or {}
        df = self.docs
        if df.empty: return []

        # FIRST: Check if this is a cuisine-type query (skip exact name search for these)
        qn = _norm(query)
        import re
        
        # List of cuisine keywords that indicate a cuisine-type query
        cuisine_query_keywords = ['indian', 'chinese', 'mexican', 'italian', 'thai', 'japanese', 'korean', 'vietnamese', 
                                   'american', 'seafood', 'pizza', 'bbq', 'steak', 'sushi', 'burger', 'taco', 'coffee',
                                   'breakfast', 'brunch', 'dessert', 'bakery', 'ice cream']
        
        is_cuisine_query = any(cuisine_kw in qn for cuisine_kw in cuisine_query_keywords)
        
        # SECOND: Try exact name search on the entire database (bypass all filters) - but skip for cuisine queries
        if len(qn) >= 3 and not is_cuisine_query:  # Skip exact name search for cuisine queries
            query_lower = qn.lower()
            
            # Extract potential restaurant names from the query
            # Look for common restaurant name patterns
            restaurant_name_patterns = []
            
            # Pattern 1: "restaurant name" + city
            city_pattern = r'(.*?)\s+(?:in|at|near)\s+(?:odessa|midland)'
            city_match = re.search(city_pattern, query_lower)
            if city_match:
                restaurant_name_patterns.append(city_match.group(1).strip())
            
            # Pattern 2: "is restaurant name open"
            open_pattern = r'(?:is|are)\s+(.*?)\s+(?:open|closed)'
            open_match = re.search(open_pattern, query_lower)
            if open_match:
                restaurant_name_patterns.append(open_match.group(1).strip())
            
            # Pattern 3: "restaurant name" + "open now"
            now_pattern = r'(.*?)\s+(?:open|closed)\s+(?:now|today)'
            now_match = re.search(now_pattern, query_lower)
            if now_match:
                restaurant_name_patterns.append(now_match.group(1).strip())
            
            # Pattern 4: "what is the rating of restaurant name"
            rating_pattern = r'(?:what is the rating of|rating of)\s+(.*?)(?:\?|$)'
            rating_match = re.search(rating_pattern, query_lower)
            if rating_match:
                restaurant_name_patterns.append(rating_match.group(1).strip())
            
            # Pattern 5: "hours for restaurant name"
            hours_pattern = r'(?:hours for|hours of)\s+(.*?)(?:\?|$)'
            hours_match = re.search(hours_pattern, query_lower)
            if hours_match:
                restaurant_name_patterns.append(hours_match.group(1).strip())
            
            # Add the original query as a fallback
            restaurant_name_patterns.append(query_lower)
            
            # Try each pattern to find exact matches
            for pattern in restaurant_name_patterns:
                if len(pattern) >= 3:  # Only for patterns with 3+ characters
                    # Create variations of the pattern to handle common naming variations
                    pattern_variations = [pattern]
                    
                    # Handle common variations
                    if 'mc donald' in pattern:
                        pattern_variations.extend(['mcdonald', 'mcdonald\'s', 'mcdonalds'])
                    elif 'mcdonald' in pattern:
                        pattern_variations.extend(['mc donald', 'mcdonald\'s', 'mcdonalds'])
                    elif 'domino' in pattern:
                        pattern_variations.extend(['domino\'s', 'dominos'])
                    elif 'pizza hut' in pattern:
                        pattern_variations.extend(['pizzahut'])
                    elif 'taco bell' in pattern:
                        pattern_variations.extend(['tacobell'])
                    elif 'burger king' in pattern:
                        pattern_variations.extend(['burgerking'])
                    elif 'chick fil a' in pattern or 'chik fil a' in pattern:
                        pattern_variations.extend(['chick-fil-a', 'chik-fil-a', 'chick fila', 'chik fila'])
                    
                    # Try each variation
                    for variation in pattern_variations:
                        # First try exact substring matches
                        exact_matches = self.docs[self.docs["name"].str.lower().str.contains(variation, case=False, na=False)]
                        
                        if not exact_matches.empty:
                            # Sort by start match first, then by rating
                            exact_matches = exact_matches.assign(
                                _start_match=exact_matches["name"].str.lower().str.startswith(variation, na=False)
                            ).sort_values(["_start_match", "rating", "review_count"], ascending=[False, False, False])
                            
                            return exact_matches.head(k)[REQUIRED_COLS].to_dict(orient="records")
                        
                        # If no exact matches, try fuzzy search
                        restaurant_names = self.docs["name"].tolist()
                        fuzzy_matches = find_fuzzy_matches(variation, restaurant_names, threshold=0.7, max_results=k)
                        
                        if fuzzy_matches:
                            # Get the fuzzy matched restaurants
                            matched_names = [match[0] for match in fuzzy_matches]
                            fuzzy_results = self.docs[self.docs["name"].isin(matched_names)]
                            
                            if not fuzzy_results.empty:
                                # Sort by fuzzy score (approximated by name order) and rating
                                fuzzy_results = fuzzy_results.sort_values(["rating", "review_count"], ascending=[False, False])
                                return fuzzy_results.head(k)[REQUIRED_COLS].to_dict(orient="records")

        # SECOND: Apply filters and do regular search
        min_stars = float(filters.get("min_stars", 0.0))
        df = df[df["rating"] >= min_stars]

        if filters.get("city"):
            allow = {str(c).lower() for c in filters["city"]}
            df = df[df["city"].astype(str).str.lower().apply(lambda x: any(a in x for a in allow))]

        if filters.get("price"):
            df = df[df["price"].isin(filters["price"])]

        # Enhanced scoring: prioritize exact name matches, then partial matches
        toks = [t for t in qn.split() if len(t) >= 2]
        
        if toks:
            # Comprehensive brand name mapping for all brands in our database
            brand_mappings = {
                # Fast Food Chains
                'mcdonald': ['mcdonald\'s', 'mcdonalds', 'mcdonald'],
                'domino': ['domino\'s', 'dominos', 'domino'],
                'starbucks': ['starbucks'],
                'pizza hut': ['pizza hut'],
                'kfc': ['kfc', 'kentucky fried chicken'],
                'subway': ['subway'],
                'taco bell': ['taco bell'],
                'burger king': ['burger king'],
                'wendy': ['wendy\'s', 'wendys', 'wendy'],
                'chick-fil-a': ['chick-fil-a', 'chik-fil-a', 'chick fila', 'chik fila'],
                'whataburger': ['whataburger'],
                'jack in the box': ['jack in the box'],
                'popeyes': ['popeyes', 'popeyes louisiana kitchen'],
                'little caesar': ['little caesar\'s', 'little caesars', 'little caesar'],
                'panda express': ['panda express', 'panda'],
                'pista express': ['pista express', 'pista'],
                'taste of india': ['taste of india', 'taste of india-permian basin'],
                'chipotle': ['chipotle', 'chipotle mexican grill'],
                'five guys': ['five guys'],
                'sonic': ['sonic', 'sonic drive-in'],
                'arby': ['arby\'s', 'arbys', 'arby'],
                'carl': ['carl\'s jr', 'carls jr', 'carl'],
                'denny': ['denny\'s', 'dennys', 'denny'],
                'ihop': ['ihop'],
                'wingstop': ['wingstop'],
                'buffalo wild': ['buffalo wild wings'],
                
                # Casual Dining
                'olive garden': ['olive garden', 'olive garden italian restaurant'],
                'red lobster': ['red lobster'],
                'applebees': ['applebees'],
                'chili': ['chili\'s', 'chilis', 'chili'],
                'outback': ['outback', 'outback steakhouse'],
                'texas roadhouse': ['texas roadhouse'],
                'longhorn': ['longhorn', 'longhorn steakhouse'],
            }
            
            # Extract brand names from query and create variations
            query_variations = [qn]
            detected_brands = []
            
            for brand_key, brand_variations in brand_mappings.items():
                if brand_key in qn:
                    detected_brands.extend(brand_variations)
                    # Create query variations
                    for variation in brand_variations:
                        if variation != brand_key:
                            query_variations.append(qn.replace(brand_key, variation))
            
            # Exact name match gets highest priority (check all variations)
            exact_match = pd.Series(False, index=df.index)
            
            # Check for exact brand matches using detected brands
            for brand in detected_brands:
                exact_match = exact_match | df["name"].str.lower().str.contains(brand, case=False, na=False)
            
            # Also check for partial brand matches in restaurant names
            for brand_key in brand_mappings.keys():
                if brand_key in qn:
                    # Look for restaurants that contain any variation of this brand
                    for variation in brand_mappings[brand_key]:
                        exact_match = exact_match | df["name"].str.lower().str.contains(variation, case=False, na=False)
            
            # Enhanced category-based scoring for cuisine types
            category_score = pd.Series(0, index=df.index)
            cuisine_keywords = {
                'indian': ['indian', 'curry', 'biryani', 'tandoori', 'indian restaurant', 'indian food', 'curry house'],
                'chinese': ['chinese', 'dim sum', 'szechuan', 'cantonese', 'chinese restaurant', 'chinese food'],
                'mexican': ['mexican', 'taco', 'burrito', 'enchilada', 'mexican restaurant', 'mexican food', 'tex-mex', 'texmex', 'taqueria'],
                'italian': ['italian', 'pasta', 'pizza', 'trattoria', 'italian restaurant', 'italian food'],
                'pizza': ['pizza', 'pizzeria', 'slice', 'pizza restaurant', 'pizza place'],
                'bbq': ['bbq', 'barbecue', 'smokehouse', 'bbq restaurant', 'barbeque'],
                'thai': ['thai', 'pad thai', 'tom yum', 'thai restaurant', 'thai food'],
                'japanese': ['japanese', 'sushi', 'ramen', 'tempura', 'japanese restaurant', 'japanese food'],
                'korean': ['korean', 'bibimbap', 'kimchi', 'korean restaurant', 'korean food'],
                'vietnamese': ['vietnamese', 'pho', 'banh mi', 'vietnamese restaurant', 'vietnamese food'],
                'american': ['american', 'burgers', 'american restaurant', 'american food'],
                'seafood': ['seafood', 'fish', 'seafood restaurant'],
                'steak': ['steak', 'steakhouse', 'steak restaurant'],
                'coffee': ['coffee', 'coffee shop', 'cafe', 'coffee house'],
                'fast food': ['fast food', 'fastfood', 'quick service'],
                'breakfast': ['breakfast', 'brunch', 'breakfast restaurant'],
                'dessert': ['dessert', 'desserts', 'ice cream', 'bakery']
            }
            
            # Check for cuisine matches and apply strong scoring
            cuisine_match_found = False
            for cuisine, keywords in cuisine_keywords.items():
                if any(kw in qn for kw in keywords):
                    cuisine_match_found = True
                    # Apply very strong category scoring (multiply by 20 for cuisine-specific queries)
                    for keyword in keywords:
                        category_score += df["categories"].str.lower().fillna("").str.count(keyword) * 20
                    break
            
            # If no specific cuisine found, apply general category scoring
            if not cuisine_match_found:
                for keyword in toks:
                    category_score += df["categories"].str.lower().fillna("").str.count(keyword)
            
            # Partial name matches
            pat = re.compile("|".join(map(re.escape, toks)))
            name_score = df["name"].str.lower().str.count(pat)
            general_category_score = df["categories"].str.lower().str.count(pat)
            
            # Combined score with exact match bonus
            score = name_score.add(general_category_score, fill_value=0)
            score = score + category_score  # Add cuisine-specific category score
            score = score + (exact_match * 10)  # Exact match gets 10x bonus
        else:
            score = pd.Series(0, index=df.index)

        df = df.assign(_score=score).sort_values(
            ["_score","rating","review_count"], ascending=[False,False,False]
        )
        
        # Fallback: if no results found, try exact name search
        if len(df) == 0:
            # Try to find restaurants with exact name matches
            exact_name_search = self.docs[self.docs["name"].str.lower().str.contains(qn, case=False, na=False)]
            if not exact_name_search.empty:
                return exact_name_search.head(k)[REQUIRED_COLS].to_dict(orient="records")
        
        # Additional fallback: if we have results but none match the query, try exact name search
        if len(df) > 0:
            # Check if any results actually match the query
            query_matches = df[df["name"].str.lower().str.contains(qn, case=False, na=False)]
            if len(query_matches) == 0:
                # Try exact name search on the entire database
                exact_name_search = self.docs[self.docs["name"].str.lower().str.contains(qn, case=False, na=False)]
                if not exact_name_search.empty:
                    return exact_name_search.head(k)[REQUIRED_COLS].to_dict(orient="records")
        
        # Check for specific restaurant queries - if exact match found, return only that restaurant
        if exact_match.any():
            exact_matches = df[exact_match.values]
            if len(exact_matches) > 0:
                # For specific restaurant queries, return only the exact matches
                return exact_matches.head(k)[REQUIRED_COLS].to_dict(orient="records")
        
        # If we have cuisine-specific results, prioritize them heavily (but still respect filters)
        if cuisine_match_found:
            # Find the detected cuisine type
            detected_cuisine = None
            for cuisine, keywords in cuisine_keywords.items():
                if any(kw in qn for kw in keywords):
                    detected_cuisine = cuisine
                    break
            
            if detected_cuisine:
                # For cuisine queries, search the filtered dataframe (already has city/rating filters applied)
                cuisine_matches = df[df["categories"].str.lower().str.contains(detected_cuisine, case=False, na=False)]
                
                # If no matches in filtered df, try searching before filters but still apply them
                if len(cuisine_matches) == 0:
                    # Search entire database for cuisine
                    cuisine_matches_all = self.docs[self.docs["categories"].str.lower().str.contains(detected_cuisine, case=False, na=False)]
                    # Now apply the same filters
                    if filters:
                        min_stars_filter = float(filters.get("min_stars", 0.0))
                        cuisine_matches_all = cuisine_matches_all[cuisine_matches_all["rating"] >= min_stars_filter]
                        if filters.get("city"):
                            allow = {str(c).lower() for c in filters["city"]}
                            cuisine_matches_all = cuisine_matches_all[cuisine_matches_all["city"].astype(str).str.lower().apply(lambda x: any(a in x for a in allow))]
                        if filters.get("price"):
                            cuisine_matches_all = cuisine_matches_all[cuisine_matches_all["price"].isin(filters["price"])]
                    cuisine_matches = cuisine_matches_all
                
                if len(cuisine_matches) > 0:
                    # Sort by score (if exists), then rating and review count
                    if "_score" in cuisine_matches.columns:
                        cuisine_matches = cuisine_matches.sort_values(["_score", "rating", "review_count"], ascending=[False, False, False])
                    else:
                        cuisine_matches = cuisine_matches.sort_values(["rating", "review_count"], ascending=[False, False])
                    return cuisine_matches.head(k)[REQUIRED_COLS].to_dict(orient="records")
        
        # If no cuisine-specific results, return regular search results
        return df.head(int(max(1,k)))[REQUIRED_COLS].to_dict(orient="records")

# ============== FAISS Review RAG (passage retrieval) ==============
_model = None
_index = None
_docstore = None

def _lazy_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def _lazy_index_and_store():
    global _index, _docstore
    if _index is None or _docstore is None:
        if not (INDEX_PATH and INDEX_PATH.exists() and DOCSTORE_PATH and DOCSTORE_PATH.exists()):
            return None, None
        import faiss  # noqa: WPS433
        if DOCSTORE_PATH.suffix == ".parquet":
            ds = pd.read_parquet(DOCSTORE_PATH)
        else:
            ds = pd.read_json(DOCSTORE_PATH, lines=True)
        _docstore = ds.reset_index(drop=True)
        _index = faiss.read_index(str(INDEX_PATH))
    return _index, _docstore

def retrieve_review_passages(question: str, k: int = 8) -> List[Dict]:
    """Return top-k review passages with metadata for LLM grounding."""
    index, store = _lazy_index_and_store()
    if index is None or store is None:
        return []
    model = _lazy_model()
    import numpy as np
    qv = model.encode([question], normalize_embeddings=True)
    if isinstance(qv, list): qv = np.array(qv)
    qv = qv.astype("float32")
    D, I = index.search(qv, k)
    out: List[Dict] = []
    for idx in I[0]:
        if int(idx) < 0: continue
        meta = store.iloc[int(idx)].to_dict()
        # normalize keys we expect: text, business, city, url, biz_rating, date
        meta.setdefault("biz_rating", meta.get("rating") or meta.get("biz_rating"))
        out.append(meta)
    return out

def build_context_block(passages: List[Dict]) -> Tuple[str, List[Tuple[int, Dict]]]:
    lines, numbered = [], []
    for i, p in enumerate(passages, start=1):
        lines.append(
            f"[{i}] {p.get('business','?')} — {p.get('city','')} — ⭐{p.get('biz_rating','?')} — {p.get('date','')}\n"
            f"{str(p.get('text','')).strip()}\nURL: {p.get('url','')}"
        )
        numbered.append((i, p))
    return "\n\n".join(lines), numbered

# ============== LLM candidate prompt (from tabular hits) ==============
def build_prompt(question: str, hits: List[Dict]) -> str:
    if not hits:
        return f"Question: {question}\n\nNo candidates."
    lines = []
    for h in hits:
        hours_info = h.get('hours', 'Hours not available')
        lines.append(f"- **{h.get('name','?')}** (⭐{float(h.get('rating',0.0)):.1f} • {h.get('price','N/A')}) — "
                     f"{h.get('categories','')} — {h.get('city','')} — Hours: {hours_info}")
    return f"Question: {question}\n\nCandidates:\n" + "\n".join(lines)
