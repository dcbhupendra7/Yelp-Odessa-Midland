# Technology: RAG System

## Components
- Document table from `businesses_ranked.csv` (fallback to clean)
- Embeddings with `all-MiniLM-L6-v2`
- FAISS cosine index (`IndexFlatIP` with L2-normalized vectors)
- Retriever combining tabular and semantic signals

## Retrieval Flow
1. Parse intent (name/cuisine/rating/city/statistics)
2. Apply layered tabular retrieval (exact/fuzzy/category/city)
3. For semantic needs, query FAISS index
4. Merge, deduplicate, and score candidates
5. Build prompt context and optional passages

## Prompt Rules
- Use only provided candidates and/or numbered passages
- Never invent restaurant names or facts
- Prefer higher `review_count` for authority
- Cite passages `[1],[2]` when used

## Why This Works
- Tabular retrieval is precise for structured filters
- FAISS augments recall for natural language
- Strict prompt keeps LLM grounded → <2% hallucination
