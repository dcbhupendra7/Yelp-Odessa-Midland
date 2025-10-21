# Streamlit Community Cloud Deployment Guide

## Prerequisites
- GitHub repository with your code
- All dependencies in requirements.txt
- .env file with API keys (for local development)

## Steps to Deploy:

### 1. Prepare Your Repository
Make sure your repo has:
- ✅ `requirements.txt` (already exists)
- ✅ `src/app.py` as main entry point
- ✅ All data files in `data/` directory
- ✅ `.env` file for API keys (will be configured in Streamlit Cloud)

### 2. Deploy to Streamlit Community Cloud

1. **Go to**: https://share.streamlit.io/
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Fill in the details**:
   - Repository: `your-username/yelp_odessa_sentiment`
   - Branch: `develop` (or `main`)
   - Main file path: `src/app.py`
   - App URL: `yelp-odessa-sentiment` (custom name)

### 3. Configure Environment Variables
In the Streamlit Cloud dashboard:
- Go to your app settings
- Add these environment variables:
  ```
  OPENAI_API_KEY=your_openai_api_key_here
  OPENAI_MODEL=gpt-4o-mini
  YELP_API_KEY=your_yelp_api_key_here
  ```

### 4. Deploy!
- Click "Deploy!"
- Wait 2-3 minutes for deployment
- Your app will be live at: `https://your-app-name.streamlit.app`

## Benefits:
- ✅ **FREE** hosting
- ✅ **Automatic deployments** on git push
- ✅ **Custom domains** available
- ✅ **Environment variables** support
- ✅ **No server management**

---

## Option 2: Railway (Alternative)

### Deploy with Railway:
1. **Go to**: https://railway.app/
2. **Connect GitHub** repository
3. **Add environment variables** in Railway dashboard
4. **Deploy automatically**

---

## Option 3: Heroku (Paid)

### Deploy with Heroku:
1. **Install Heroku CLI**
2. **Create Procfile**: `web: streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0`
3. **Deploy**: `git push heroku develop:main`

---

## Important Notes:

### Data Files
- Your `data/` directory will be included in deployment
- Make sure all processed files are committed to git
- Large files (>100MB) might need Git LFS

### API Keys
- **Never commit** `.env` file to git
- Use Streamlit Cloud's environment variables instead
- Keep your API keys secure

### Performance
- Streamlit Cloud has resource limits
- For heavy usage, consider paid options
- Your app should work fine for moderate usage

## Quick Start (Recommended):
1. Push your latest changes: `git push origin develop`
2. Go to https://share.streamlit.io/
3. Connect your GitHub repo
4. Set main file to `src/app.py`
5. Add environment variables
6. Deploy!

Your app will be live in minutes! 🎉
