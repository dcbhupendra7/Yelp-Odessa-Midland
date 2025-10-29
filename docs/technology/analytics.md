# Technology: Advanced Analytics

## Ranking
- Bayesian weighted rating with global mean `C` and threshold `m`
- Popularity via `log1p(review_count)`
- `rank_score = bayes_score * (1 + 0.15 * popularity)`

## Investor Insights Analytics
- Market Opportunity: filter categories with `avg_rating ≥ 4.0` and `business_count < 5`; score = `(avg_rating * avg_review_count) / (business_count + 1)`
- Location Hotspots: KMeans clustering on lat/lng; per-cluster stats (dominant city, avg rating, density)
- Competitor Benchmark: avg/median rating, review volume, count, most common price tier

## Data Cleaning Highlights
- Coerce numeric columns; fill missing strings; ensure required columns exist
- Transform categories into list for filtering/exploding

## Visualizations
- Plotly charts (histograms, bars)
- PyDeck map with color-coded clusters and tooltips
