# Results: Performance Metrics

## Query Latency

| Search Type | Average | P95 |
|---|---:|---:|
| Exact name match | <10 ms | 15 ms |
| Fuzzy name match | <50 ms | 80 ms |
| FAISS vector search | <100 ms | 200 ms |
| Full query (with LLM) | <2 s | 5 s |

## Pipeline Times (Local CPU)

| Step | Avg Time |
|---|---:|
| Full data collection (two cities, 36+ categories) | ~18 min |
| Business metrics (ranked CSV) | ~2 min |
| RAG index build | ~5 min |

## Dataset Stats

- Restaurants: ~1,200+
- Reviews (count field sum): ~31,000+
- Categories covered: 36+

## Automation
- Daily refresh via GitHub Actions
- ~95% workflow reliability observed (failures mainly from API rate-limits)
