"""
Dashboard Configuration
"""

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 5

# Dashboard Settings
PAGE_TITLE = "Telco Customer Intelligence Platform"
PAGE_ICON = "🎯"
LAYOUT = "wide"

# Data Paths
DATA_PATH = "data/processed/processed_telco_data.csv"
SEGMENTS_PATH = "data/processed/customer_segments.csv"
SEGMENT_SUMMARY_PATH = "data/processed/segment_summary.csv"

# Business Parameters
DEFAULT_CLV = 2000
DEFAULT_INTERVENTION_COST = 25
DEFAULT_SUCCESS_RATE = 0.3
DEFAULT_ACQUISITION_COST = 500

# Refresh Intervals (seconds)
METRICS_REFRESH = 300  # 5 minutes
PREDICTIONS_REFRESH = 60  # 1 minute

# Color Schemes
RISK_COLORS = {
    "Low Risk": "#00cc00",
    "Medium Risk": "#ffa500",
    "High Risk": "#ff6b6b",
    "Critical": "#cc0000",
}

CHART_COLORS = {
    "primary": "#3498db",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#3498db",
}

# Feature Lists
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Dashboard Pages
PAGES = {
    "🏠 Executive Dashboard": "executive",
    "🔮 Real-time Predictions": "predictions",
    "📊 Customer Analytics": "analytics",
    "💰 ROI Calculator": "roi",
    "🎯 Segmentation": "segmentation",
    "⚙️ What-If Scenarios": "scenarios",
    "📈 Model Performance": "performance",
}
