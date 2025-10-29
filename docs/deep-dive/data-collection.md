# Deep Dive: Data Collection (yelp_fetch_reviews.py)

## Purpose
Fetch business-only data from Yelp for Odessa and Midland across many categories, with resumable caching and periodic CSV writes.

## Key Design Choices
- Cities: Odessa, TX and Midland, TX (override via `--cities`)
- Categories: 36+ cuisine aliases (override via file)
- Pagination: 50 results/page up to configurable `--max_offset`
- Throttling: Sleep between calls (default 0.25s)
- Resumable: Per-page JSON cache + manifest to skip already-fetched pages

## File Paths
- Cache root: `data/cache/<City>_<State>/<category>/<offset>.json`
- Manifest: `data/cache/manifest.json`
- Raw CSV: `data/raw/businesses.csv`
- Clean CSV: `data/processed/businesses_clean.csv`

## CLI Usage
```bash
python src/yelp_fetch_reviews.py \
  --sleep 0.25 \
  --max_offset 1000 \
  --cities "Odessa, TX" "Midland, TX" \
  --save_every 200
```

## High-Level Flow
1. Load categories and cities
2. Load manifest; ensure cache directories exist
3. For each city × category × offset:
   - If cached page exists (per manifest), load from cache
   - Else call Yelp `/businesses/search` and persist page JSON
   - Normalize fields and append to in-memory batch
   - Every N pages, flush batch to CSVs
4. Final flush to CSV; update manifest

## Normalized Columns
- id, name, rating, review_count, price, categories (titles joined by comma)
- address, city, state, zip_code
- latitude, longitude
- url
- hours (computed from business_hours; human-readable string)

## Hours Formatting
Incoming Yelp `business_hours[0].open[]` is converted to per-day ranges (e.g., "Monday: 11:00-14:30 | 16:30-21:00").

## Resumable Caching
- Cache key: `<city>||<category>||<offset>`
- Manifest tracks fetched keys; reruns skip previously cached pages
- Each page JSON is saved independently to allow true resume

## Error Handling & Backoff
- Retries on 429/5xx with exponential backoff
- Non-200 responses cache an empty array to avoid hot loops

## Outputs
- `data/raw/businesses.csv` (all rows)
- `data/processed/businesses_clean.csv` (unique by `id`)

## Environment
- Requires `YELP_API_KEY` in env or `.env`

## Extensibility
- Add cities or categories via flags/file
- Adjust save cadence with `--save_every`
- Integrate additional fields by expanding the `keep` list in the normalizer
