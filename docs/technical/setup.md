# Setup Guide

## 🚀 Quick Start

Get the platform running in minutes.

---

## Prerequisites

- **Python 3.8+** (3.12 recommended)
- **Yelp API Key** ([Get one here](https://www.yelp.com/developers))
- **OpenAI API Key** (optional, for enhanced chat features)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/yelp_odessa_sentiment.git
cd yelp_odessa_sentiment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Required for Yelp data fetching
YELP_API_KEY=your_yelp_api_key_here

# Optional for enhanced chat features
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 4. Run Data Pipeline

```bash
# Step 1: Fetch Yelp data
python src/yelp_fetch_reviews.py

# Step 2: Process and rank businesses
python src/prepare_business_metrics.py

# Step 3: Build RAG index
python src/build_rag_index.py
```

### 5. Launch Application

```bash
streamlit run src/app.py
```

Access at: **http://localhost:8501**

---

## Alternative: Automated Setup

For GitHub Actions automation, see [Deployment Guide](deployment.md).

---

## Troubleshooting

### Missing API Keys

**Error:** `Missing YELP_API_KEY`

**Solution:** 
1. Check your `.env` file exists
2. Verify API key is correct
3. Make sure file is in project root

### Missing Data Files

**Error:** `Missing processed CSV`

**Solution:** Run the data pipeline (Step 4 above)

### Port Already in Use

**Error:** `Port 8501 is already in use`

**Solution:**
```bash
streamlit run src/app.py --server.port 8502
```

---

## Verification

After setup, verify everything works:

1. ✅ Analytics page loads with data
2. ✅ Chat page responds to queries
3. ✅ Investor Insights shows opportunities
4. ✅ Maps display restaurant locations

---

For detailed troubleshooting, see [Technical Documentation](api.md).

