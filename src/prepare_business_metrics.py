#!/usr/bin/env python3
"""
Create ranked business metrics (Bayesian weighted rating with popularity)
Input:  data/processed/businesses_clean.csv
Output: data/processed/businesses_ranked.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

INP = Path("data/processed/businesses_clean.csv")
OUT = Path("data/processed/businesses_ranked.csv")

def bayesian_weighted_rating(R, v, C, m):
    # IMDb-like formula: (v/(v+m))*R + (m/(v+m))*C
    return (v/(v+m))*R + (m/(v+m))*C

def main():
    if not INP.exists():
        raise SystemExit("Missing businesses_clean.csv. Run yelp_fetch_reviews.py first.")
    df = pd.read_csv(INP)

    # Global mean stars (C) and min reviews (m)
    C = df["rating"].mean()
    # choose m as 60th percentile of review_count (controls popularity effect)
    m = float(df["review_count"].quantile(0.60))

    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).clip(lower=0)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).clip(lower=0, upper=5)

    df["bayes_score"] = bayesian_weighted_rating(df["rating"], df["review_count"], C, m)
    df["popularity"]  = np.log1p(df["review_count"])           # softer scaling
    df["rank_score"]  = df["bayes_score"] * (1 + 0.15 * df["popularity"])  # tunable blend

    df.sort_values(["rank_score","bayes_score","review_count","rating"], ascending=False, inplace=True)
    df.to_csv(OUT, index=False)
    print(f"✅ Saved ranked businesses -> {OUT} (rows: {len(df):,})")

if __name__ == "__main__":
    main()
