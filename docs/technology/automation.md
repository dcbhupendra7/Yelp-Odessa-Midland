# Technology: Automation

## Auto Refresh (src/auto_refresh_data.py)

### Modes
- `check`: freshness/integrity report
- `full`: backup → fetch → process → build RAG → update metadata
- `incremental`: throttled (≥6h) full-refresh without backup
- `--setup-cron`: writes a cron line for daily updates

### Metadata
- `processed/data_metadata.json` stores last refresh, counts, history

### Integrity Checks
- Verifies required CSV files exist and are non-empty
- Reports row counts and file sizes

## Scheduling
- Cron line example: daily at 2 AM
- Can be run from GitHub Actions (self-hosted workflow) if needed

## Docs Deployment
- `.github/workflows/mkdocs.yml` builds and publishes documentation to GitHub Pages on push

## Backups
- `data/backups/<timestamp>/` with processed CSVs and metadata
