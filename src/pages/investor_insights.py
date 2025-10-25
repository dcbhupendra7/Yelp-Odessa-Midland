#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from sklearn.cluster import KMeans
import plotly.express as px

PROC = Path("data/processed")
CSV_RANKED = PROC / "businesses_ranked.csv"
CSV_CLEAN  = PROC / "businesses_clean.csv"

@st.cache_data(show_spinner=False)
def load_businesses() -> pd.DataFrame:
    """Load restaurant data for investor analysis, filtered to Odessa and Midland only."""
    if CSV_RANKED.exists():
        df = pd.read_csv(CSV_RANKED)
    elif CSV_CLEAN.exists():
        df = pd.read_csv(CSV_CLEAN)
    else:
        st.error("No business data found. Please run data processing first.")
        st.stop()
    
    # Filter to only Odessa and Midland restaurants (handle case variations)
    df = df[df['city'].str.lower().isin(['odessa', 'midland'])]
    
    # Clean missing values
    df = df.dropna(subset=['latitude', 'longitude', 'rating', 'review_count'])
    return df

def clean_categories(categories_str):
    """Clean and split categories into a list of cuisine tags."""
    if pd.isna(categories_str):
        return []
    
    # Split on common separators and clean
    categories = str(categories_str).replace('/', ',').split(',')
    return [cat.strip() for cat in categories if cat.strip()]

def compute_market_opportunity(df, city):
    """
    Identify cuisine categories with high ratings but low competition.
    Returns opportunities sorted by opportunity score.
    """
    # Filter by city if specified
    if city != "Both":
        df_filtered = df[df['city'].str.contains(city, case=False, na=False)]
    else:
        df_filtered = df
    
    # Explode categories to get individual cuisine types
    df_filtered = df_filtered.copy()
    df_filtered['category_list'] = df_filtered['categories'].apply(clean_categories)
    df_exploded = df_filtered.explode('category_list')
    
    # Remove empty categories
    df_exploded = df_exploded[df_exploded['category_list'].str.len() > 0]
    
    # Group by category and compute metrics
    category_stats = df_exploded.groupby('category_list').agg({
        'rating': ['mean', 'count'],
        'review_count': 'mean',
        'name': 'count'
    }).round(2)
    
    # Flatten column names
    category_stats.columns = ['avg_rating', 'rating_count', 'avg_review_count', 'business_count']
    
    # Filter for high-quality, low-competition categories
    opportunities = category_stats[
        (category_stats['avg_rating'] >= 4.0) & 
        (category_stats['business_count'] < 5)
    ].copy()
    
    # Calculate opportunity score
    opportunities['opportunity_score'] = (
        opportunities['avg_rating'] * 
        opportunities['avg_review_count'] / 
        (opportunities['business_count'] + 1)
    )
    
    # Sort by opportunity score
    opportunities = opportunities.sort_values('opportunity_score', ascending=False)
    
    return opportunities.head(10)

def cluster_locations(df):
    """
    Cluster restaurants geographically to identify location hotspots.
    Returns cluster analysis with metrics per cluster.
    """
    # Prepare data for clustering
    coords = df[['latitude', 'longitude']].values
    
    # Use KMeans clustering
    n_clusters = min(4, len(df))  # Use 4 clusters or fewer if not enough data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(coords)
    
    # Calculate cluster statistics
    def get_dominant_city(x):
        mode_result = x.mode()
        return mode_result.iloc[0] if len(mode_result) > 0 else 'Mixed'
    
    cluster_stats = df.groupby('cluster_id').agg({
        'city': get_dominant_city,
        'rating': 'mean',
        'review_count': 'mean',
        'name': 'count',
        'latitude': 'mean',
        'longitude': 'mean'
    }).round(2)
    
    cluster_stats.columns = ['dominant_city', 'avg_rating', 'avg_review_count', 'business_count', 'center_lat', 'center_lng']
    cluster_stats = cluster_stats.sort_values('avg_rating', ascending=False)
    
    return df, cluster_stats

def benchmark_competitors(df, cuisine, city):
    """
    Analyze competition for a specific cuisine in a specific city.
    Returns competitor metrics and strategic insights.
    """
    # Filter data
    df_filtered = df[df['city'].str.contains(city, case=False, na=False)]
    
    # Find businesses that serve this cuisine
    cuisine_businesses = df_filtered[
        df_filtered['categories'].str.contains(cuisine, case=False, na=False)
    ]
    
    if len(cuisine_businesses) == 0:
        return None, f"No {cuisine} restaurants found in {city}."
    
    # Calculate metrics
    avg_rating = cuisine_businesses['rating'].mean()
    median_rating = cuisine_businesses['rating'].median()
    avg_review_count = cuisine_businesses['review_count'].mean()
    competitor_count = len(cuisine_businesses)
    
    # Most common price tier
    price_counts = cuisine_businesses['price'].value_counts()
    most_common_price = price_counts.index[0] if len(price_counts) > 0 else "N/A"
    
    metrics = {
        'avg_rating': avg_rating,
        'median_rating': median_rating,
        'avg_review_count': avg_review_count,
        'competitor_count': competitor_count,
        'most_common_price': most_common_price
    }
    
    # Strategic insight
    insight = f"If you open a new {cuisine} restaurant in {city}, you'll be competing mostly at the {most_common_price} price point. The average rating you need to beat to stand out is about {avg_rating:.1f}★, and the market currently only has {competitor_count} direct competitors."
    
    return metrics, insight

# Main page
st.title("💰 Investor Insights: Odessa & Midland Only")
st.markdown("**Strategic analysis for restaurant investment opportunities in Odessa & Midland only**")

# Load data
df = load_businesses()

# Section 1: Market Opportunity (Category Gap)
st.header("🎯 Market Opportunity Analysis")
st.markdown("**Identify cuisine categories with high customer satisfaction but low competition**")

# City filter
city_option = st.selectbox(
    "Select city for analysis:",
    ["Both", "Odessa", "Midland"],
    help="Choose which city to analyze for market opportunities"
)

# Compute opportunities
opportunities = compute_market_opportunity(df, city_option)

if len(opportunities) > 0:
    st.markdown(f"**These categories have high ratings (4.0+ stars) but low competition (<5 businesses) in {city_option}. They look like strong opportunities for a new business.**")
    
    # Display opportunities table
    st.dataframe(
        opportunities,
        use_container_width=True,
        column_config={
            "avg_rating": st.column_config.NumberColumn("Avg Rating", format="%.2f"),
            "avg_review_count": st.column_config.NumberColumn("Avg Reviews", format="%.0f"),
            "business_count": st.column_config.NumberColumn("Competitors", format="%.0f"),
            "opportunity_score": st.column_config.NumberColumn("Opportunity Score", format="%.2f")
        }
    )
else:
    st.info(f"No clear opportunities found in {city_option} with current criteria. Consider lowering the rating threshold or increasing competitor count.")

# Section 2: Location Hotspots
st.header("📍 Location Hotspots")
st.markdown("**Identify geographic areas with high restaurant performance but room for growth**")

# Cluster analysis
df_with_clusters, cluster_stats = cluster_locations(df)

# Display cluster table
st.markdown("**Cluster Analysis - Ranked by Average Rating:**")
st.dataframe(
    cluster_stats,
    use_container_width=True,
    column_config={
        "avg_rating": st.column_config.NumberColumn("Avg Rating", format="%.2f"),
        "avg_review_count": st.column_config.NumberColumn("Avg Reviews", format="%.0f"),
        "business_count": st.column_config.NumberColumn("Restaurants", format="%.0f"),
        "center_lat": st.column_config.NumberColumn("Latitude", format="%.4f"),
        "center_lng": st.column_config.NumberColumn("Longitude", format="%.4f")
    }
)

# Strategic insight for top cluster
if len(cluster_stats) > 0:
    top_cluster = cluster_stats.index[0]
    top_city = cluster_stats.iloc[0]['dominant_city']
    top_rating = cluster_stats.iloc[0]['avg_rating']
    top_count = cluster_stats.iloc[0]['business_count']
    
    st.markdown(f"**Strategic Insight:** Cluster {top_cluster} in {top_city} has a high average rating of {top_rating:.1f}★ but only {top_count:.0f} restaurants. This may be a target zone for expansion.")

# Map visualization
st.markdown("**Geographic Distribution by Cluster:**")
st.markdown("*Restaurants are colored by cluster ID. Hover for details.*")

# Create map
map_data = df_with_clusters[['latitude', 'longitude', 'cluster_id', 'name', 'rating', 'price']].copy()
map_data['cluster_id'] = map_data['cluster_id'].astype(str)

# Color mapping for clusters
def get_cluster_color(cluster_id):
    colors = {
        '0': [255, 0, 0],      # Red
        '1': [0, 255, 0],      # Green  
        '2': [0, 0, 255],      # Blue
        '3': [255, 255, 0]     # Yellow
    }
    return colors.get(str(cluster_id), [128, 128, 128])

map_data['color'] = map_data['cluster_id'].apply(get_cluster_color)

layer = pdk.Layer(
    'ScatterplotLayer',
    data=map_data,
    get_position='[longitude, latitude]',
    get_color='color',
    get_radius=100,
    pickable=True,
    tooltip={
        'html': '<b>{name}</b><br/>Rating: {rating}★<br/>Price: {price}<br/>Cluster: {cluster_id}',
        'style': {'backgroundColor': 'steelblue', 'color': 'white'}
    }
)

view_state = pdk.ViewState(
    latitude=map_data['latitude'].mean(),
    longitude=map_data['longitude'].mean(),
    zoom=10,
    pitch=0
)

st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=view_state,
    layers=[layer]
))

# Section 3: Competitor Benchmark
st.header("⚔️ Competitor Benchmark")
st.markdown("**Understand the competitive landscape for specific cuisines and cities**")

col1, col2 = st.columns(2)

with col1:
    # Get unique cuisines
    all_categories = []
    for categories in df['categories'].dropna():
        all_categories.extend(clean_categories(categories))
    
    unique_cuisines = sorted(list(set(all_categories)))
    selected_cuisine = st.selectbox(
        "Select cuisine/category:",
        unique_cuisines,
        help="Choose the type of restaurant you're considering"
    )

with col2:
    selected_city = st.selectbox(
        "Select city:",
        ["Odessa", "Midland"],
        help="Choose the target city"
    )

# Compute competitor analysis
metrics, insight = benchmark_competitors(df, selected_cuisine, selected_city)

if metrics:
    st.markdown("**Competitive Analysis:**")
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Rating",
            f"{metrics['avg_rating']:.1f}★",
            help="Rating you need to beat to stand out"
        )
    
    with col2:
        st.metric(
            "Typical Price Tier",
            metrics['most_common_price'],
            help="Most common price point in this market"
        )
    
    with col3:
        st.metric(
            "Competitor Count",
            f"{metrics['competitor_count']:.0f}",
            help="Number of direct competitors"
        )
    
    with col4:
        st.metric(
            "Avg Review Volume",
            f"{metrics['avg_review_count']:.0f}",
            help="Typical review count for this cuisine"
        )
    
    # Strategic insights
    st.markdown("**Strategic Insights:**")
    st.markdown(f"• {insight}")
    st.markdown(f"• Median rating in this market: {metrics['median_rating']:.1f}★")
    st.markdown(f"• Average review volume: {metrics['avg_review_count']:.0f} reviews per restaurant")
    
    if metrics['competitor_count'] < 3:
        st.success("🟢 **Low Competition**: Few competitors means easier market entry")
    elif metrics['competitor_count'] < 6:
        st.warning("🟡 **Moderate Competition**: Balanced market with room for differentiation")
    else:
        st.error("🔴 **High Competition**: Many competitors - focus on unique value proposition")

else:
    st.warning(insight)

# Footer
st.markdown("---")
st.markdown("*Analysis based on Yelp data for Odessa & Midland restaurants. Data as of latest collection.*")
