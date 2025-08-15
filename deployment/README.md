# 🚀 Telco Customer Intelligence Platform - Deployment Guide

## Overview
This comprehensive guide covers all deployment options for the Telco Customer Intelligence Platform, from local development to cloud production environments.

## Quick Start

### Prerequisites
- Python 3.9+
- Required data files in `data/processed/`
- API running on `http://localhost:8000`

### Minimal Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the platform
python scripts/launch_all.py
```

## 🏠 Option 1: Local Deployment

### Basic Local Setup
```bash
# Start dashboard only
streamlit run src/dashboard/app.py

# Access at: http://localhost:8501
```

### Advanced Local Setup with API
```bash
# Terminal 1: Start API
python src/api/main.py

# Terminal 2: Start Dashboard
streamlit run src/dashboard/app.py --server.port 8501
```

### Performance-Optimized Local
```bash
# Use the comprehensive launcher
python scripts/launch_all.py

# Features:
# - Automatic dependency checking
# - Health monitoring
# - Graceful shutdown
# - Browser auto-launch
```

---

## 🐳 Option 2: Docker Deployment

### Quick Docker Start
```bash
# Start dashboard service only
docker-compose up dashboard

# Start full platform
docker-compose up -d
```

### Production Docker Setup
```bash
# Build optimized images
docker-compose build

# Start with scaling
docker-compose up -d --scale api=3 --scale dashboard=2

# Monitor
docker-compose logs -f
docker-compose ps
```

### Docker Environment Variables
Create `.env` file:
```bash
# Database
DB_PASSWORD=your_secure_password
DB_HOST=postgres
DB_PORT=5432

# API Configuration
API_URL=http://api:8000

# Dashboard Configuration
DASHBOARD_PORT=8501
```

---

## ☁️ Option 3: Streamlit Cloud Deployment

### Step 1: Prepare Repository
```bash
# Ensure your code is in GitHub
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main
```

### Step 2: Create Streamlit Cloud App
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set main file path: `src/dashboard/app.py`

### Step 3: Configure Environment Variables
In Streamlit Cloud settings, add:
```
API_URL = "https://your-api-domain.com"
STREAMLIT_THEME_PRIMARY_COLOR = "#3498db"
STREAMLIT_THEME_BACKGROUND_COLOR = "#ffffff"
```

### Step 4: Advanced Streamlit Cloud Configuration
Create `requirements.txt` for cloud:
```
streamlit==1.48.0
plotly==5.15.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.28.0
xlsxwriter>=3.0.0
```

---

## 🌐 Option 4: Heroku Deployment

### Step 1: Create Heroku Files

#### Procfile
```bash
web: streamlit run src/dashboard/app.py --server.port=$PORT --server.address=0.0.0.0
```

#### runtime.txt
```
python-3.9.18
```

#### requirements.txt (Heroku-specific)
```
streamlit==1.48.0
plotly==5.15.0
pandas>=2.0.0
numpy>=1.24.0
requests>=2.28.0
xlsxwriter>=3.0.0
gunicorn>=20.1.0
```

### Step 2: Heroku Configuration
```bash
# Install Heroku CLI
# Then:
heroku create your-telco-dashboard
heroku config:set API_URL="https://your-api-app.herokuapp.com"
git push heroku main
```

### Step 3: Heroku Environment Setup
```bash
# Set environment variables
heroku config:set STREAMLIT_THEME_PRIMARY_COLOR="#3498db"
heroku config:set STREAMLIT_THEME_BACKGROUND_COLOR="#ffffff"
heroku config:set STREAMLIT_SERVER_ENABLE_CORS=false
```

---

## 🌟 Option 5: AWS Deployment

### EC2 Deployment
```bash
# On EC2 instance
sudo yum update -y
sudo yum install python3 python3-pip -y
pip3 install -r requirements.txt

# Start with PM2 for process management
pm2 start "streamlit run src/dashboard/app.py" --name telco-dashboard
```

### ECS Deployment
Use the provided `Dockerfile.dashboard` with ECS task definition.

---

## 📈 Performance Optimization Implementation

### 1. Advanced Caching Strategy
```python
import streamlit as st
import pandas as pd

# In src/dashboard/app.py - already optimized
@st.cache_data(ttl=300, max_entries=3)
def load_data():
    """Cached data loading with TTL"""
    return pd.read_csv("data/processed/processed_telco_data.csv")

@st.cache_data(ttl=600)
def load_model_predictions():
    """Cache expensive model operations"""
    # Example function definition
    def expensive_computation():
        """Example function for expensive computation"""
        return {"predictions": [0.1, 0.2, 0.3]}
    
    return expensive_computation()
```

### 2. Session State Management
```python
import streamlit as st

# Initialize session state for better UX
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None
    
if 'selected_filters' not in st.session_state:
    st.session_state.selected_filters = {}
```

### 3. Lazy Loading Implementation
```python
import streamlit as st
import pandas as pd

# Example function definitions
def compute_advanced_analytics():
    """Example function for advanced analytics computation"""
    return {"correlations": {}, "insights": []}

def display_advanced_analytics(analytics):
    """Example function to display analytics"""
    st.json(analytics)

# Load heavy components only when needed
if st.button("Show Advanced Analytics"):
    with st.spinner("Loading advanced analytics..."):
        if 'advanced_analytics' not in st.session_state:
            st.session_state.advanced_analytics = compute_advanced_analytics()
        
        display_advanced_analytics(st.session_state.advanced_analytics)
```

---

## 🔧 Configuration Files for Each Deployment

### Streamlit Cloud: `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#3498db"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200
```

### Heroku: `setup.sh` (if needed)
```bash
#!/bin/bash
mkdir -p ~/.streamlit/
echo "[server]
port = $PORT
enableCORS = false
headless = true
" > ~/.streamlit/config.toml
```

---

## 🚀 Deployment Checklist

### Pre-Deployment
- [ ] All tests passing (`python scripts/test_dashboard_integration.py`)
- [ ] Data files prepared
- [ ] Environment variables configured
- [ ] Dependencies verified

### Local Deployment
- [ ] Virtual environment activated
- [ ] API running (if separate)
- [ ] Dashboard accessible at localhost:8501

### Docker Deployment
- [ ] Docker installed and running
- [ ] Images built successfully
- [ ] All services healthy
- [ ] Ports accessible

### Cloud Deployment
- [ ] Repository pushed to GitHub
- [ ] Environment variables set
- [ ] Domain configured (if custom)
- [ ] SSL certificates (if needed)

---

## 📊 Monitoring and Maintenance

### Health Checks
```python
import streamlit as st
import pandas as pd
from datetime import datetime

# Example function definition
def check_api_connection():
    """Example function to check API connection"""
    return True

# Built into the dashboard
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "data_loaded": not st.session_state.get('data', pd.DataFrame()).empty,
        "api_connected": check_api_connection()
    }
```

### Performance Monitoring
- Monitor response times
- Track memory usage
- Monitor user sessions
- Check API connectivity

### Maintenance Tasks
- Regular data updates
- Model retraining
- Dependency updates
- Security patches

---

## 🆘 Troubleshooting

### Common Issues
1. **Port Conflicts**: Change port in config
2. **Memory Issues**: Increase container resources
3. **API Connection**: Check network connectivity
4. **Data Loading**: Verify file paths

### Debug Commands
```bash
# Check logs
streamlit run src/dashboard/app.py --logger.level=debug

# Docker debugging
docker-compose logs dashboard
docker-compose exec dashboard bash

# Health check
curl http://localhost:8501/_stcore/health
```

---

## 🎯 Production Best Practices

1. **Security**: Use HTTPS, environment variables for secrets
2. **Performance**: Enable caching, optimize queries
3. **Monitoring**: Set up health checks, logging
4. **Backup**: Regular data backups, code versioning
5. **Scaling**: Load balancing, horizontal scaling

Choose the deployment option that best fits your infrastructure and requirements!
