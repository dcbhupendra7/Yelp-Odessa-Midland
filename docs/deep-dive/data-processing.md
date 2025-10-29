# Deep Dive: Data Processing & RAG Indexing

## Part A — Business Metrics (prepare_business_metrics.py)

### Goal
Create ranked metrics for restaurants using Bayesian weighted rating and popularity blending.

### Inputs / Outputs
- Input: `data/processed/businesses_clean.csv`
- Output: `data/processed/businesses_ranked.csv`

### Steps
1. Coerce `rating` and `review_count` to numeric and clamp valid ranges
2. Compute global mean rating `C`
3. Set popularity threshold `m` = 60th percentile of `review_count`
4. Bayesian weighted rating:
   - `(v/(v+m))*R + (m/(v+m))*C`
5. Popularity = `log1p(review_count)`
6. Rank score = `bayes_score * (1 + 0.15 * popularity)`
7. Sort by `rank_score`, then `bayes_score`, `review_count`, `rating`

### Why Bayesian?
Balances high ratings with sample size so a 5.0 with 1 review doesn’t dominate a 4.5 with 500 reviews.

---

## Part B — RAG Index (build_rag_index.py)

### Goal
Build a FAISS index over business cards for fast semantic retrieval in chat.

### Inputs / Outputs
- Input: `businesses_ranked.csv` (fallback to `businesses_clean.csv`)
- Outputs:
  - `data/processed/rag/faiss.index`
  - `data/processed/rag/docstore.parquet` (records with text)
  - `data/processed/rag/meta.json` (model metadata)

### Document Construction
Each row becomes a textual card with:
- Name, categories, price tier, stars (with review count), full address, URL

### Embeddings & Index
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Normalize vectors (L2)
- FAISS `IndexFlatIP` (cosine via normalized dot-product)

### Why This Design
- Small, fast, well-known model with good recall/latency balance
- Flat index is sufficient for dataset size; trivial to swap for IVF/HNSW later

---

## Data Quality
- Missing `price` → "N/A"; missing text fields → ""; numeric coercion with defaults
- Unique by `id` in clean CSV

## Rebuilding
- Re-run both scripts after new fetch or on schedule via auto-refresh
