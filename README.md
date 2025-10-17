Of course. Based on the files you've provided for your project, here is a comprehensive `README.md` file that you can use.

---

# 🍽️ Yelp Odessa & Midland Analytics + RAG Chat

A comprehensive Streamlit application for analyzing Yelp business data from Odessa and Midland, Texas, featuring interactive analytics, geographic visualization, and an AI-powered restaurant recommendation chat system.

## 🌟 Features

### 📊 Analytics Dashboard

- **Interactive Visualizations**: Charts and graphs for business metrics.
- **Geographic Mapping**: Interactive map showing restaurant locations.
- **Advanced Filtering**: Filter by city, price range, rating, and review count.
- **Business Rankings**: Employs Bayesian weighted ratings to provide more accurate rankings.
- **Data Export**: Functionality to export filtered results to a CSV file.
- **KPI Metrics**: Displays key performance indicators and aggregated statistics.

### 💬 AI-Powered Chat Assistant

- **RAG (Retrieval-Augmented Generation)**: Delivers context-aware restaurant recommendations.
- **Natural Language Queries**: Ask questions like "best pizza in Odessa" or "worst rated tacos".
- **Multi-Intent Support**: Can handle complex queries with multiple restaurant types.
- **Brand Detection**: Recognizes specific restaurant chains and brands.
- **Filtered Search**: Applies city, price, and rating filters to chat responses.
- **GPT-4o-mini Integration**: Enhanced responses with OpenAI's language model (optional).

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Yelp API Key ([Get one here](https://www.yelp.com/developers))
- OpenAI API Key (optional, for enhanced chat features)

### Installation

1.  **Clone the repository**

    ```bash
    git clone <repository-url>
    cd yelp_odessa_sentiment
    ```

2.  **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables**
    Create a `.env` file in the project root with your API keys:

    ```bash
    # Required for Yelp data fetching
    YELP_API_KEY=your_yelp_api_key_here

    # Optional for enhanced chat features
    OPENAI_API_KEY=your_openai_api_key_here
    OPENAI_MODEL=gpt-4o-mini
    ```

4.  **Run the data pipeline**
    Before launching the app, you need to fetch and process the data:

    ```bash
    python src/yelp_fetch_reviews.py
    python src/prepare_business_metrics.py
    python src/build_rag_index.py
    ```

5.  **Run the application**

    ```bash
    streamlit run src/app.py
    ```

6.  **Access the app**
    Open your browser and navigate to: **http://localhost:8502**

## 📁 Project Structure

```
yelp_odessa_sentiment/
├── src/
│   ├── app.py                    # Main Streamlit application
│   ├── yelp_fetch_reviews.py     # Yelp API data fetcher
│   ├── prepare_business_metrics.py # Business ranking processor
│   ├── build_rag_index.py        # RAG index builder
│   ├── pages/
│   │   ├── analytics.py          # Analytics dashboard page
│   │   └── chat.py               # RAG chat interface
│   └── utils/
│       ├── llm_openai.py         # OpenAI integration
│       └── rag.py                # RAG retrieval system
├── data/
│   ├── raw/                      # Raw Yelp data
│   ├── processed/                # Processed business data
│   ├── cache/                    # API response cache
│   └── rag/                      # RAG index files
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Technical Details

### Dependencies

- **Streamlit**: Web application framework
- **Pandas & NumPy**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **PyDeck**: Geographic mapping
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **OpenAI**: Language model integration
- **Requests**: API communication
- **python-dotenv**: Environment variable management

### Data Sources

- **Yelp Fusion API**: Business data and reviews.
- **Categories**: Fetches data for over 40 restaurant categories, including Mexican, Italian, Pizza, Burgers, BBQ, Chinese, Coffee, Seafood, Steak, Sushi, and many more.

### Algorithms

- **Bayesian Weighted Rating**: IMDb-style rating calculation to balance ratings with the number of reviews.
- **Popularity Scoring**: Logarithmic scaling of review counts to factor in popularity.
- **Vector Search**: Cosine similarity for semantic search in the RAG system.
