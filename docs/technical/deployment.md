# Technical: Deployment

## Streamlit App

### Local
```bash
streamlit run src/app.py
```

### Streamlit Cloud
1. Push repo to GitHub
2. Create app at Streamlit Cloud → point to `src/app.py`
3. Set secrets:
   - `YELP_API_KEY`
   - `OPENAI_API_KEY` (optional)
4. Deploy

## Automated Data Refresh (GitHub Actions)
- Use the provided workflow in `.github/workflows/mkdocs.yml` for docs
- Add a similar workflow (not included here) to run `auto_refresh_data.py --mode incremental` on schedule if needed

## MkDocs Documentation → GitHub Pages

1. Confirm `mkdocs.yml` has your `site_url`, repo links
2. Ensure workflow `.github/workflows/mkdocs.yml` exists
3. GitHub → Settings → Pages → Branch: `gh-pages` (root)
4. Push to `main`; docs auto-deploy to:
```
https://dcbhupendra7.github.io/yelp_odessa_sentiment/
```

## Environment & Secrets
- Keep API keys in environment (local `.env`) or GitHub Secrets (CI)
- Never commit secrets to the repo

## Data Storage
- Generated data under `data/` folders; excluded by `.gitignore`

## Rollback
- Use backups under `data/backups/`
- Rebuild index with `build_rag_index.py` after restore
