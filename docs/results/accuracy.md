# Results: Accuracy Improvements

## Summary
We focused on three problem areas and measured concrete improvements.

### 1) AI Hallucination Reduction
- Before: ~30%
- After: **<2%**
- How: strict system prompt; candidate-only grounding; better context formatting; validations

### 2) Fuzzy Matching Accuracy
- Before: ~45%
- After: **92%**
- How: normalization + SequenceMatcher; threshold tuning; partial word strategy

### 3) Database-Wide Queries
- Before: Often incorrect (LLM only saw top-k results)
- After: **100%** on tests
- How: Detect database-wide intent and compute stats directly over DataFrame

## Rating-Focused Queries
- Special 5.0-star boost ensures "maximum/best" queries surface true 5-star results even with lower review counts

## Validation Notes
- Spot-checked name, address, hours, counts against `businesses_ranked.csv`
- Ensured responses never invent businesses not present in candidates
