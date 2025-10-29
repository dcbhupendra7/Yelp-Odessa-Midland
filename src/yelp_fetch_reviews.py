#!/usr/bin/env python3
"""
Resumable Yelp business fetcher (business-only, no reviews)
- Two cities (Odessa, TX & Midland, TX) by default
- Many categories (edit below or pass a file)
- Paginates up to Yelp's limit per query
- Caches each page so reruns are instant (resume)
- Periodically writes partial CSVs

Usage (defaults are safe):
  python src/yelp_fetch_reviews.py

Optional flags:
  --sleep 0.25              # seconds between calls
  --max_offset 1000         # up to Yelp limit for /search (multiples of 50)
  --categories_file path    # newline-separated category aliases
  --cities "Odessa, TX" "Midland, TX"
  --save_every 200          # flush partial CSV every N pages
"""

from __future__ import annotations
import os
import time
import json
import argparse
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm

# Load .env file if it exists (for local development)
# In GitHub Actions, environment variables are set directly
load_dotenv()

# --------- Config (defaults) ----------
DEFAULT_CITIES = ["Odessa, TX", "Midland, TX"]
DEFAULT_CATEGORIES = [
    "mexican","italian","pizza","burgers","bbq","sandwiches","chinese","coffee",
    "seafood","steak","sushi","thai","indian","breakfast_brunch","vegan","desserts",
    "icecream","salad","pubs","bars","fastfood","mediterranean","noodles","korean",
    "vietnamese","cajun","tacos","bakery","foodtrucks","grill","soulfood","buffets",
    "diners","chicken_wings","ramen","poke","tex-mex"
]
RESULTS_PER_QUERY = 50   # Yelp max per page
DEFAULT_MAX_OFFSET = 1000
DEFAULT_SLEEP = 0.25     # seconds (keeps us under 50 req/min comfortably)
DEFAULT_SAVE_EVERY = 200 # write partial CSV every N pages

# --------- Paths ----------
RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR = Path("data/processed"); PROC_DIR.mkdir(parents=True, exist_ok=True)
OUT_RAW = RAW_DIR / "businesses.csv"
OUT_CLEAN = PROC_DIR / "businesses_clean.csv"

CACHE_DIR = Path("data/cache")     # per-page JSON cache
MANIFEST = CACHE_DIR / "manifest.json"  # tracks fetched pages

# --------- API ----------
API_KEY = os.getenv("YELP_API_KEY")
if not API_KEY:
    raise SystemExit("❌ Missing YELP_API_KEY. Set it as environment variable or in .env file")
BASE_URL = "https://api.yelp.com/v3/businesses/search"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    ap.add_argument("--max_offset", type=int, default=DEFAULT_MAX_OFFSET)
    ap.add_argument("--save_every", type=int, default=DEFAULT_SAVE_EVERY)
    ap.add_argument("--categories_file", type=str, default=None)
    ap.add_argument("--cities", nargs="*", default=None, help='Override cities, e.g. --cities "Odessa, TX" "Midland, TX"')
    return ap.parse_args()


def load_categories(path: str | None) -> List[str]:
    if not path:
        return DEFAULT_CATEGORIES
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Categories file not found: {p}")
    return [line.strip() for line in p.read_text().splitlines() if line.strip() and not line.startswith("#")]


def load_manifest() -> Dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text())
        except Exception:
            pass
    return {"fetched": {}}  # key: f"{city}||{cat}||{offset}": 1


def save_manifest(man: Dict[str, Any]) -> None:
    MANIFEST.write_text(json.dumps(man, indent=2))


def cache_key(city: str, cat: str, offset: int) -> str:
    return f"{city}||{cat}||{offset}"


def cache_path(city: str, cat: str, offset: int) -> Path:
    # safe filenames
    safe_city = city.replace(", ", "_").replace(" ", "_")
    safe_cat = cat.replace(" ", "_")
    d = CACHE_DIR / safe_city / safe_cat
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{offset:05d}.json"


def get_page(session: requests.Session, city: str, cat: str, offset: int, sleep_s: float) -> List[Dict[str, Any]]:
    """Fetch one Yelp search page; uses cache if present."""
    p = cache_path(city, cat, offset)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass

    params = {
        "location": city,
        "categories": cat,
        "limit": RESULTS_PER_QUERY,
        "offset": offset,
        "sort_by": "rating",
    }
    # Basic retry/backoff
    for attempt in range(5):
        r = session.get(BASE_URL, headers=HEADERS, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json().get("businesses", [])
            p.write_text(json.dumps(data))
            time.sleep(sleep_s)
            return data
        elif r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** attempt * 0.5, 8.0))
            continue
        else:
            # cache empty to avoid hot loop on bad query
            p.write_text(json.dumps([]))
            return []
    return []


def flatten(raw_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not raw_rows:
        return pd.DataFrame()
    df = pd.json_normalize(raw_rows)
    keep = [
        "id","name","rating","review_count","price","categories",
        "location.address1","location.city","location.state","location.zip_code",
        "coordinates.latitude","coordinates.longitude","url","business_hours"
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.rename(columns={
        "location.address1": "address",
        "location.city": "city",
        "location.state": "state",
        "location.zip_code": "zip_code",
        "coordinates.latitude": "latitude",
        "coordinates.longitude": "longitude",
    }, inplace=True)
    df["categories"] = df["categories"].apply(
        lambda x: ", ".join([c["title"] for c in x]) if isinstance(x, list) else (x or "")
    )
    
    # Process business hours into readable format
    df["hours"] = df["business_hours"].apply(format_hours)
    df.drop("business_hours", axis=1, inplace=True)
    
    # Coerce numerics
    for col in ["rating","review_count","latitude","longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def format_hours(hours_data):
    """Convert Yelp hours data to readable format."""
    if not hours_data or not isinstance(hours_data, list) or len(hours_data) == 0:
        return "Hours not available"
    
    # Get the first hours entry (usually there's only one)
    hours_entry = hours_data[0]
    if not hours_entry.get("open"):
        return "Hours not available"
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    formatted_hours = []
    
    # Group hours by day
    day_hours = {}
    for time_slot in hours_entry["open"]:
        day = time_slot["day"]
        start = time_slot["start"]
        end = time_slot["end"]
        
        # Convert HHMM to readable time
        start_time = f"{start[:2]}:{start[2:]}"
        end_time = f"{end[:2]}:{end[2:]}"
        
        if day not in day_hours:
            day_hours[day] = []
        day_hours[day].append(f"{start_time}-{end_time}")
    
    # Format each day
    for day_num in range(7):
        if day_num in day_hours:
            day_name = days[day_num]
            hours_str = ", ".join(day_hours[day_num])
            formatted_hours.append(f"{day_name}: {hours_str}")
        else:
            formatted_hours.append(f"{days[day_num]}: Closed")
    
    return " | ".join(formatted_hours)


def flush_to_csv(parts: List[pd.DataFrame]) -> None:
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT_RAW, index=False)
    clean = (
        df.drop_duplicates("id")
          .reset_index(drop=True)
    )
    clean.to_csv(OUT_CLEAN, index=False)


def main():
    args = parse_args()

    cities = args.cities or DEFAULT_CITIES
    categories = load_categories(args.categories_file)
    max_offset = max(0, args.max_offset)
    if max_offset % RESULTS_PER_QUERY != 0:
        max_offset = (max_offset // RESULTS_PER_QUERY) * RESULTS_PER_QUERY

    session = requests.Session()
    manifest = load_manifest()
    fetched = manifest.get("fetched", {})

    page_counter = 0
    batch_frames: List[pd.DataFrame] = []

    print(f"📡 Fetching: {len(cities)} cities × {len(categories)} categories; up to {max_offset} offset per query.")
    for city in cities:
        for cat in tqdm(categories, desc=f"{city}", leave=False):
            for offset in range(0, max_offset, RESULTS_PER_QUERY):
                key = cache_key(city, cat, offset)
                if key in fetched:
                    # we have this page already (cache/manifest)
                    data = json.loads(cache_path(city, cat, offset).read_text())
                else:
                    data = get_page(session, city, cat, offset, args.sleep)
                    fetched[key] = 1
                    manifest["fetched"] = fetched
                    # persist manifest frequently to allow true resume
                    if page_counter % 10 == 0:
                        save_manifest(manifest)

                if not data:
                    # no more results for this query; move to next category
                    break

                df = flatten(data)
                if not df.empty:
                    df["fetched_city"] = city
                    df["fetched_category"] = cat
                    batch_frames.append(df)

                page_counter += 1
                if page_counter % args.save_every == 0:
                    flush_to_csv(batch_frames)

    # final flush
    flush_to_csv(batch_frames)
    save_manifest(manifest)

    if OUT_CLEAN.exists():
        df_final = pd.read_csv(OUT_CLEAN)
        print(f"✅ Done. Unique businesses: {len(df_final):,}")
        print(f"   Raw cache: {str(CACHE_DIR)}")
        print(f"   Raw CSV:   {str(OUT_RAW)}")
        print(f"   Clean CSV: {str(OUT_CLEAN)}")
    else:
        print("⚠️ No businesses saved. Check API key or network.")


if __name__ == "__main__":
    main()
