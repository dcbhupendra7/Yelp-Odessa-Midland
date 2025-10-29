# Technical Glossary (Plain English)

A quick reference for terms used across the app and documentation.

---

## Data & Columns
- **rating**: Yelp star rating (0.0–5.0).
- **review_count**: Total number of Yelp reviews for a business.
- **price**: Price tier — `$` (cheap) to `$$$$` (expensive); `None`/`N/A` means unknown.
- **categories**: Comma‑separated list of cuisines/tags from Yelp.
- **city / address / zip_code / latitude / longitude**: Standard location fields.
- **hours**: Human‑readable business hours built from Yelp’s raw schedule.

---

## Analytics & Scoring
- **Bayesian weighted rating (bayes_score)**: A rating that balances a restaurant’s own stars with the global average to avoid over‑promoting places with very few reviews. Formula: `(v/(v+m))*R + (m/(v+m))*C` where `R`=restaurant rating, `v`=review_count, `C`=global mean, `m`=popularity threshold.
- **popularity**: `log1p(review_count)` — softer growth so very large counts don’t dominate.
- **rank_score**: Overall ranking score used in tables — `bayes_score * (1 + 0.15 * popularity)`.
- **reliability_score**: Used for “best” queries — `rating * log(review_count + 1)`; prefers highly‑rated places with more reviews.
- **Opportunity Score**: For investor analysis — `(avg_rating × avg_review_count) ÷ (business_count + 1)`; higher is better (quality, visibility, low competition).

---

## Clustering & Geography
- **KMeans**: An algorithm that groups nearby points into clusters by distance.
- **cluster_id (0,1,2,3)**: The numeric label assigned by KMeans. It has no inherent meaning by itself — it’s just “group 0/1/2/3”. We compute per‑cluster stats (average rating, business_count, center_lat/lng) to interpret them.
- **cluster center (center_lat/center_lng)**: The average latitude/longitude of restaurants in the cluster — a “hotspot” center.

---

## Retrieval & AI
- **RAG (Retrieval‑Augmented Generation)**: The model answers using only information we retrieve from our dataset (candidates and optional passages), preventing hallucination.
- **embeddings**: Numeric vectors representing text meaning. We use `all‑MiniLM‑L6‑v2`.
- **FAISS**: A fast vector index. We build it over restaurant text cards for semantic search.
- **Index type (IndexFlatIP)**: Cosine similarity via dot product after L2 normalization.
- **candidates**: The set of restaurant rows retrieved by our tabular/semantic search before the model answers.
- **grounding**: Forcing the AI to use only the supplied candidates/passages.

---

## Search Strategy (Tabular Layer)
- **Exact match**: Direct string match on business names.
- **Fuzzy match**: Tolerates typos using similarity (e.g., “mcdonalds” → “McDonald’s”).
- **Cuisine match**: Looks for cuisine keywords in `categories` (e.g., mexican, ramen, sushi).
- **City filter**: Restricts to Odessa or Midland.
- **Rating filter**: `rating ≥ X` from queries like “rating more than 3”.
- **Price filter**: Filters by `$`, `$$`, `$$$`, `$$$$`.
- **Fallbacks**: If a strict query fails, we progressively relax filters to always give a data‑grounded answer.

---

## Data Pipeline
- **Data Collection**: Fetch pages from Yelp, cache JSON per city/category/offset, and write CSVs.
- **Processed CSVs**: `businesses_clean.csv`, `businesses_ranked.csv` (with ranking columns).
- **RAG Index Files**: `rag/faiss.index`, `rag/docstore.parquet`, `rag/meta.json`.
- **Backups**: Timestamped copies under `data/backups/` before refresh.

---

## Automation & Freshness
- **Auto refresh**: `auto_refresh_data.py` — full or incremental update; validates files and updates `data_metadata.json`.
- **Hot‑reload (chat)**: The chat’s retriever automatically reloads the latest table if the file changes, so answers reflect daily updates.
- **GitHub Actions**: CI that can deploy docs and (optionally) run refresh workflows on schedule.

---

## UI & Maps
- **CARTO provider**: Basemap used for PyDeck maps without requiring a Mapbox token.
- **Tooltips**: On map hover, we show `name`, `city`, `rating`, `price`, and `cluster label`.
- **Legend**: Colors (red/green/blue/yellow) map to cluster IDs 0–3 for quick scanning.

---

## Common Phrases in the Docs
- **“High rating, low competition”**: Categories with `avg_rating ≥ 4.0` and `business_count < 5`.
- **“Database‑wide stats”**: Questions like “how many 5‑star restaurants?” that are answered by scanning the full table, not just search hits.
- **“Grounded responses”**: Answers built only from the retrieved rows/passages — nothing invented.
