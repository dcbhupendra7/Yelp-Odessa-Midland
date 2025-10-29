# Technical: Script Interfaces (APIs/CLI)

## yelp_fetch_reviews.py

```bash
python src/yelp_fetch_reviews.py \
  --sleep 0.25 \
  --max_offset 1000 \
  --categories_file categories.txt \
  --cities "Odessa, TX" "Midland, TX" \
  --save_every 200
```

- Reads `YELP_API_KEY` from environment
- Outputs raw and clean CSVs as described in Deep Dive → Data Collection

## prepare_business_metrics.py

```bash
python src/prepare_business_metrics.py
```

- Input: `data/processed/businesses_clean.csv`
- Output: `data/processed/businesses_ranked.csv`

## build_rag_index.py

```bash
python src/build_rag_index.py
```

- Creates FAISS index + docstore under `data/processed/rag/`

## auto_refresh_data.py

```bash
# Status report
python src/auto_refresh_data.py --mode check

# Full refresh (with backup by default)
python src/auto_refresh_data.py --mode full

# Incremental update (throttled to ≥6h)
python src/auto_refresh_data.py --mode incremental

# Generate cron line
python src/auto_refresh_data.py --setup-cron
```

### Outputs
- Backups under `data/backups/<timestamp>/`
- `processed/data_metadata.json` with counts and history

## Streamlit App

```bash
streamlit run src/app.py
```

- Pages: Analytics, Chat, Investor Insights

## Environment Variables
- `YELP_API_KEY` (required)
- `OPENAI_API_KEY` (optional for Chat)
- Optional overrides:
  - `DATA_DIR`, `RAG_DOC_TABLE`, `RAG_INDEX_PATH`, `RAG_DOCSTORE_PATH`, `EMBED_MODEL`
