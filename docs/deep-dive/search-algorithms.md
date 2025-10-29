# Deep Dive: Search Algorithms (utils/rag.py)

## Overview
The retriever combines multiple strategies to maximize accuracy and robustness for restaurant queries.

## Strategies (Applied in Order)
1. Exact name match (fast path)
2. Fuzzy name match (SequenceMatcher with threshold ~0.6)
3. Cuisine/category match
4. Rating-focused search (prioritize 5.0 when asking for "best" or "maximum rating")
5. City-based filtering (Odessa vs Midland)
6. Partial word/substring match
7. Semantic vector retrieval (FAISS) as needed
8. Fallback: top-ranked by `rank_score`

## Normalization
- Lowercasing, punctuation stripping, whitespace collapsing
- Typo tolerance via fuzzy similarity

## Scoring
- Base score + boosts:
  - 5.0-star boost for "maximum/best" rating queries
  - Extra weight for higher `review_count`

## Required Columns
The retriever ensures presence of: `name, url, rating, review_count, city, address, categories, price, latitude, longitude, id, hours` (missing ones filled with defaults).

## Prompt Context (for LLM)
- Candidate cards list with name, rating, price, reviews, address, categories, hours
- Optional FAISS passages for citations
- System rules: never invent businesses; use only provided info; cite when using passages

## Why Multi-Strategy?
- Real users type typos and partial names
- Some queries are categorical ("best sushi in Odessa")
- Others are statistical ("how many 5-star?")
- Layered approach yields >95% overall accuracy in our tests
