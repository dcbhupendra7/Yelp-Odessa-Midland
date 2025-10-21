#!/usr/bin/env python3

import re
import unicodedata

# Copy the parsing functions from chat.py
LIMIT_RE = re.compile(r"(?:show|limit|top|first)\s+(\d+)", re.I)
CITY_RE = re.compile(r"\b(odessa|midland)\b", re.I)
STARS_RE = re.compile(r"(\d(?:\.\d)?)\s*\+?\s*stars?", re.I)
PRICE_RE = re.compile(r"\$|\$\$|\$\$\$|\$\$\$\$|price\s*[:=]?\s*(none|n/a)", re.I)

BEST_WORDS = {"best", "top", "highest", "greatest", "excellent", "outstanding", "amazing", "fantastic"}
WORST_WORDS = {"worst", "lowest", "terrible", "awful", "bad", "poor", "horrible"}
AVG_WORDS = {"average", "mean", "typical", "normal"}
FEW_REVIEWS = {"few", "little", "low", "minimal"}

def parse_limit(q, default_k): 
    m = LIMIT_RE.search(q); 
    return max(1,min(20,int(m.group(2)))) if m else default_k

def parse_cities(q): 
    return list({m.group(1).title() for m in CITY_RE.finditer(q)})

def parse_min_stars(q, d): 
    m = STARS_RE.search(q); 
    return float(m.group(1)) if m else d

def parse_prices(q):
    out = []; 
    for m in PRICE_RE.finditer(q):
        t = m.group(0).lower()
        out.append("None" if "none" in t or "n/a" in t else "$$$$" if "$$$$" in t else "$$$" if "$$$" in t else "$$" if "$$" in t else "$")
    res = []; seen = set()
    for p in out:
        if p not in seen: res.append(p); seen.add(p)
    return res

def intent_kind(q):
    qn = q.lower()
    if any(w in qn for w in BEST_WORDS): return "best"
    if any(w in qn for w in WORST_WORDS): return "worst"
    if any(w in qn for w in AVG_WORDS): return "average"
    if any(w in qn for w in FEW_REVIEWS): return "few_reviews"
    return "generic"

# Test the parsing functions
if __name__ == "__main__":
    query = 'suggest me best panda express in odessa'

    print('=== TESTING PARSING FUNCTIONS ===')
    print(f'Query: "{query}"')
    print()

    cities = parse_cities(query)
    print(f'Cities parsed: {cities}')

    k = parse_limit(query, 8)
    print(f'Limit parsed: {k}')

    min_stars = parse_min_stars(query, 0.0)
    print(f'Min stars parsed: {min_stars}')

    prices = parse_prices(query)
    print(f'Prices parsed: {prices}')

    kind = intent_kind(query)
    print(f'Intent kind: {kind}')

    print()
    print('Filters that would be applied:')
    filters = {'min_stars': min_stars}
    if cities: filters['city'] = cities
    if prices: filters['price'] = prices
    print(f'Filters: {filters}')
