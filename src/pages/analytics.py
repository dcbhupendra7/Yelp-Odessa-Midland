#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

PROC = Path("data/processed")
CSV_RANKED = PROC / "businesses_ranked.csv"
CSV_CLEAN  = PROC / "businesses_clean.csv"

@st.cache_data(show_spinner=False)
def load_businesses() -> pd.DataFrame:
    if CSV_RANKED.exists():
        df = pd.read_csv(CSV_RANKED)
    elif CSV_CLEAN.exists():
        df = pd.read_csv(CSV_CLEAN)
    else:
        st.error("Missing processed CSV. Run:\n\n"
                 "  python src/yelp_fetch_reviews.py\n"
                 "  python src/prepare_business_metrics.py\n"
                 "  python src/build_rag_index.py")
        st.stop()
    df["price"] = df["price"].fillna("None")
    df["categories"] = df["categories"].fillna("")
    df["city"] = df["city"].fillna("")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(0, 5)
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
    df["latitude"]  = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["categories_list"] = df["categories"].map(lambda s: [c.strip() for c in s.split(",") if c.strip()])
    return df

st.title("📊 Analytics")

df_all = load_businesses()

with st.sidebar:
    st.header("Filters")
    st.markdown("[📚 Documentation](https://dcbhupendra7.github.io/Yelp-Odessa-Midland/)")
    sel_cities  = st.multiselect("City", ["Odessa", "Midland"], [])
    sel_prices  = st.multiselect("Price", ["$", "$$", "$$$", "$$$$", "None"], [])
    min_stars   = st.slider("Min stars", 0.0, 5.0, 0.0, 0.5)
    min_reviews = st.number_input("Min Yelp review count", value=0, min_value=0, step=10)
    st.caption("These filters apply to this page.")
    st.info("💡 **Tip:** Set min stars to 3.5+ to focus on higher-rated restaurants only.")

mask = (df_all["rating"] >= float(min_stars)) & (df_all["review_count"] >= int(min_reviews))
if sel_prices:
    mask &= df_all["price"].isin(sel_prices)
if sel_cities:
    mask &= df_all["city"].isin(sel_cities)
df = df_all[mask].copy()

# KPI row
k1, k2, k3, k4 = st.columns(4)
with k1: st.markdown(f"<div class='kpi'><h3>Businesses</h3><p>{df['id'].nunique():,}</p></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='kpi'><h3>Avg Stars</h3><p>{(df['rating'].mean() if not df.empty else 0):.2f}</p></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='kpi'><h3>Median Stars</h3><p>{(df['rating'].median() if not df.empty else 0):.2f}</p></div>", unsafe_allow_html=True)
with k4: st.markdown(f"<div class='kpi'><h3>Total Yelp Reviews</h3><p>{int(df['review_count'].sum()):,}</p></div>", unsafe_allow_html=True)

st.divider()

# Left: Ratings hist | Right: Price bars
c1, c2 = st.columns([2, 1], gap="large")
with c1:
    st.subheader("Ratings Distribution")
    if not df.empty:
        fig = px.histogram(df, x="rating", nbins=24, height=360, color_discrete_sequence=["#ff6f61"])
        fig.update_layout(xaxis_title="Yelp Stars", yaxis_title="Businesses", bargap=0.06)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for current filters.")
with c2:
    st.subheader("Businesses by Price")
    if not df.empty:
        tmp = (df.groupby("price", dropna=False)
                 .agg(businesses=("id","nunique"), mean_rating=("rating","mean"))
                 .reset_index())
        order = {"None":0,"$":1,"$$":2,"$$$":3,"$$$$":4}
        tmp["order"] = tmp["price"].map(order).fillna(0)
        tmp = tmp.sort_values("order")
        fig2 = px.bar(tmp, x="price", y="businesses", hover_data=["mean_rating"], height=360,
                      color_discrete_sequence=["#4aa3ff"])
        fig2.update_layout(xaxis_title="Price (tiers)", yaxis_title="Businesses")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data for current filters.")

st.subheader("Mean Rating by Category (Filtered)")
df_ex = df.explode("categories_list").rename(columns={"categories_list":"category"})
grp = (df_ex.groupby("category", dropna=True)
       .agg(businesses=("id","nunique"), mean_rating=("rating","mean"), median_rating=("rating","median"))
       .reset_index()
       .sort_values(["businesses","mean_rating"], ascending=[False, False]))
cL, cR = st.columns([3,1])
with cR:
    min_biz = st.slider("Min businesses per category", 1, 30, 3, 1)
    top_n   = st.slider("Top N categories", 5, 50, 25, 5)
viz = grp[grp["businesses"] >= min_biz].head(top_n)
if not viz.empty:
    fig3 = px.bar(viz, x="category", y="mean_rating",
                  hover_data=["businesses","median_rating"], height=480,
                  color_discrete_sequence=["#8bdc65"])
    fig3.update_layout(xaxis_title="Category", yaxis_title="Mean Stars")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No categories meet the threshold.")

st.subheader("Map")
map_df = df[["name","rating","review_count","price","latitude","longitude","categories","address","city"]].dropna(subset=["latitude","longitude"])
if not map_df.empty:
    color_by_rating = (map_df["rating"] - 3.0).clip(0, 2) / 2.0
    map_df = map_df.assign(_color=(color_by_rating * 255).astype(int))
    deck = pdk.Deck(
        map_provider="carto",
        map_style="light",
        initial_view_state=pdk.ViewState(
            latitude=float(map_df["latitude"].mean()),
            longitude=float(map_df["longitude"].mean()),
            zoom=11,
            pitch=0,
            bearing=0
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[longitude, latitude]",
                get_radius=80,
                get_fill_color="[32+_color*0.0, 136+_color*0.0, 255-_color*0.3]",
                pickable=True,
                opacity=0.8,
            )
        ],
        tooltip={
            "html": "<b>{name}</b><br/>📍 {city}<br/>⭐ {rating} • {review_count} reviews • {price}<br/>{categories}",
            "style": {"backgroundColor": "#ffffff", "color": "#000000", "fontSize": "13px", "padding": "6px", "border": "1px solid #ccc"}
        }
    )
    st.pydeck_chart(deck)
else:
    st.info("No lat/long with current filters.")

st.subheader("Businesses (Filtered)")
cols = ["name","city","rating","review_count","price","address","categories","url"]
if not df.empty:
    st.dataframe(df[cols].sort_values(["rating","review_count"], ascending=[False, False]),
                 use_container_width=True, height=420)
    st.download_button("Download filtered CSV", df[cols].to_csv(index=False),
                       "yelp_businesses_filtered.csv", "text/csv")
else:
    st.info("No rows match filters.")
