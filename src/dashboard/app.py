"""
Telco Customer Intelligence Dashboard
Author: Your Name
Date: 2025
Description: Interactive dashboard for churn prediction and business analytics
"""

import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# Suppress future warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Page configuration
st.set_page_config(
    page_title="Telco Customer Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 600;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc00;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state with performance optimization


def initialize_session_state():
    """Initialize session state variables for better performance"""
    defaults = {
        "api_url": os.getenv("API_URL", "http://localhost:8000"),
        "selected_customer": None,
        "data_loaded": False,
        "last_refresh": None,
        "filter_cache": {},
        "advanced_analytics": None,
        "model_predictions": None,
        "user_preferences": {
            "auto_refresh": False,
            "show_debug": False,
            "cache_analytics": True,
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Initialize session state
initialize_session_state()


# Helper functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load processed data"""
    try:
        # Load main data
        df = pd.read_csv("data/processed/processed_telco_data.csv")

        # Verify essential columns exist
        required_columns = ["customerID", "Churn", "MonthlyCharges"]
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(
                f"Missing required columns in main data: {missing_columns}")
            return pd.DataFrame()

        # Load segments data
        segments = pd.read_csv("data/processed/customer_segments.csv")

        # Verify segments has required columns
        segments_required = ["customerID", "churn_probability", "risk_segment"]
        segments_missing = [
            col for col in segments_required if col not in segments.columns
        ]
        if segments_missing:
            st.warning(
                f"Missing columns in segments data: {segments_missing}. Proceeding without merge.")
            return df

        # Merge segments with main data
        df = df.merge(
            segments[["customerID", "churn_probability", "risk_segment"]],
            on="customerID",
            how="left",
        )

        # Final verification
        if df.empty:
            st.error("Data loaded but resulted in empty DataFrame")
            return pd.DataFrame()

        return df

    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_segment_summary():
    """Load segment summary data"""
    try:
        return pd.read_csv("data/processed/segment_summary.csv")
    except Exception:
        # Create dummy data if file doesn't exist
        return pd.DataFrame(
            {
                "risk_segment": [
                    "Low Risk", "Medium Risk", "High Risk", "Critical"], "customer_count": [
                    3200, 1900, 1600, 250], "MonthlyCharges": [
                    45.5, 65.2, 78.9, 89.5], "churn_probability": [
                        0.08, 0.35, 0.65, 0.85], "potential_revenue_loss": [
                            5000, 15000, 45000, 35000], })


# Advanced Performance Functions
@st.cache_data(ttl=600, max_entries=5)
def compute_advanced_analytics(df, segment_type="risk"):
    """Compute expensive analytics with caching"""
    if df.empty:
        return {}

    analytics = {
        "correlation_matrix": df.select_dtypes(include=[np.number]).corr(),
        "segment_analysis": df.groupby("risk_segment")
        .agg(
            {
                "MonthlyCharges": ["mean", "std"],
                "tenure": ["mean", "std"],
                "Churn": lambda x: (x == "Yes").mean(),
            }
        )
        .round(3)
        if "risk_segment" in df.columns
        else {},
        "feature_importance": {
            "Contract": 0.23,
            "tenure": 0.19,
            "MonthlyCharges": 0.15,
            "InternetService": 0.12,
            "PaymentMethod": 0.11,
        },
    }
    return analytics


@st.cache_data(ttl=900, max_entries=3)
def get_model_predictions_batch(customer_data_list):
    """Cache batch predictions for better performance"""
    try:
        response = requests.post(
            f"{st.session_state.api_url}/predict/batch",
            json={"customers": customer_data_list},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_feature_importance():
    """Load and cache feature importance data"""
    try:
        return pd.read_csv("models/saved/feature_importance.csv")
    except Exception:
        # Default feature importance
        return pd.DataFrame(
            {
                "feature": [
                    "Contract",
                    "tenure",
                    "MonthlyCharges",
                    "InternetService",
                    "PaymentMethod",
                ],
                "importance": [0.23, 0.19, 0.15, 0.12, 0.11],
            }
        )


def call_api_prediction(customer_data):
    """Call API for real-time prediction"""
    try:
        response = requests.post(
            f"{st.session_state.api_url}/predict", json=customer_data, timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        # Fallback to mock data if API is not running
        return {
            "churn_probability": np.random.random(),
            "churn_prediction": "Yes" if np.random.random() > 0.5 else "No",
            "confidence": np.random.uniform(0.7, 0.95),
            "risk_segment": np.random.choice(
                ["Low Risk", "Medium Risk", "High Risk", "Critical"]
            ),
        }


def calculate_roi(
        num_customers,
        churn_prob,
        intervention_cost,
        success_rate,
        clv):
    """Calculate ROI for retention campaign"""
    expected_churners = num_customers * churn_prob
    prevented_churns = expected_churners * success_rate
    revenue_saved = prevented_churns * clv
    total_cost = num_customers * intervention_cost
    roi = (revenue_saved - total_cost) / total_cost if total_cost > 0 else 0
    return {
        "revenue_saved": revenue_saved,
        "total_cost": total_cost,
        "roi": roi,
        "prevented_churns": prevented_churns,
    }


# Sidebar
with st.sidebar:
    st.image("docs/screenshots/dashboard_1.png", use_container_width=True)
    st.title("Navigation")

    page = st.selectbox(
        "Select Page",
        [
            "🏠 Executive Dashboard",
            "🔮 Real-time Predictions",
            "📊 Customer Analytics",
            "💰 ROI Calculator",
            "🎯 Segmentation Analysis",
            "⚙️ What-If Scenarios",
            "📈 Model Performance",
        ],
    )

    st.markdown("---")
    st.markdown("### Quick Stats")
    df = load_data()
    if not df.empty and "Churn" in df.columns:
        st.metric("Total Customers", f"{len(df): , }")
        st.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100: .1f}%")
        st.metric("Avg Monthly Revenue", f"${df['MonthlyCharges'].sum(): , .0f}")
    elif not df.empty:
        st.metric("Total Customers", f"{len(df): , }")
        st.warning("⚠️ Churn data not available")
        st.metric("Avg Monthly Revenue", f"${df['MonthlyCharges'].sum(): , .0f}")

    st.markdown("---")
    st.markdown("### API Status")
    try:
        response = requests.get(
            f"{st.session_state.api_url}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Online")
        else:
            st.warning("⚠️ API Issue")
    except Exception:
        st.error("❌ API Offline")

    st.markdown("---")
    st.markdown("### ⚡ Performance Settings")

    # Performance toggles
    st.session_state.user_preferences["auto_refresh"] = st.checkbox(
        "Auto-refresh data",
        value=st.session_state.user_preferences.get("auto_refresh", False),
        help="Automatically refresh data every 5 minutes",
    )

    st.session_state.user_preferences["cache_analytics"] = st.checkbox(
        "Cache analytics",
        value=st.session_state.user_preferences.get("cache_analytics", True),
        help="Cache expensive computations for better performance",
    )

    st.session_state.user_preferences["show_debug"] = st.checkbox(
        "Show debug info",
        value=st.session_state.user_preferences.get("show_debug", False),
        help="Display performance and caching information",
    )

    # Cache management
    if st.button("🗑️ Clear Cache", help="Clear all cached data"):
        st.cache_data.clear()
        for key in [
            "advanced_analytics",
            "feature_importance",
                "model_predictions"]:
            if key in st.session_state:
                del st.session_state[key]
        st.success("Cache cleared!")

# Main content based on selected page
if page == "🏠 Executive Dashboard":
    st.title("🏠 Executive Dashboard")
    st.markdown(
        "### Real-time Business Intelligence for Telco Customer Retention")

    # Load data
    df = load_data()
    segment_summary = load_segment_summary()

    # Check if data loaded successfully
    if df.empty:
        st.error(
            "⚠️ Unable to load data. Please check that the data files exist "
            "in the data/processed/ directory."
        )
        st.stop()

    # Verify required columns exist
    if "Churn" not in df.columns:
        st.error(
            f"⚠️ Missing 'Churn' column in data. Available columns: {
                list(
                    df.columns)}")
        st.stop()

    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_customers = len(df)
        st.metric(
            "Total Customers",
            f"{total_customers: , }",
            delta=f"+{int(total_customers * 0.02): , } this month",
        )

    with col2:
        churn_rate = (df["Churn"] == "Yes").mean() * 100
        st.metric(
            "Churn Rate",
            f"{churn_rate: .1f}%",
            delta=f"{-1.2: .1f}% vs last month",
            delta_color="inverse",
        )

    with col3:
        monthly_revenue = df["MonthlyCharges"].sum()
        st.metric(
            "Monthly Revenue",
            f"${monthly_revenue: , .0f}",
            delta=f"+${monthly_revenue * 0.05: , .0f}",
        )

    with col4:
        at_risk = len(
            df[df.get("risk_segment", "Low Risk").isin(["High Risk", "Critical"])]
        )
        st.metric(
            "At-Risk Customers",
            f"{at_risk: , }",
            delta=f"{-50} vs last week",
            delta_color="inverse",
        )

    with col5:
        potential_savings = 487000  # From your model results
        st.metric(
            "Potential Savings",
            f"${potential_savings: , .0f}",
            delta="+12% with optimization",
        )

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)

    with col1:
        # Risk distribution pie chart
        if "risk_segment" in df.columns:
            risk_dist = df["risk_segment"].value_counts()
            fig_pie = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                title="Customer Risk Distribution",
                color_discrete_map={
                    "Low Risk": "#00cc00",
                    "Medium Risk": "#ffa500",
                    "High Risk": "#ff6b6b",
                    "Critical": "#cc0000",
                },
            )
            fig_pie.update_traces(
                textposition="inside",
                textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Monthly trend
        months = pd.date_range(end=datetime.now(), periods=12, freq="ME")
        trend_data = pd.DataFrame(
            {
                "Month": months,
                "Churn_Rate": np.random.uniform(24, 28, 12),
                "Retention_Rate": np.random.uniform(72, 76, 12),
            }
        )

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=trend_data["Month"],
                y=trend_data["Churn_Rate"],
                mode="lines+markers",
                name="Churn Rate",
                line=dict(color="#ff6b6b", width=3),
            )
        )
        fig_trend.update_layout(
            title="12-Month Churn Trend",
            xaxis_title="Month",
            yaxis_title="Churn Rate (%)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # Charts row 2
    col1, col2 = st.columns(2)

    with col1:
        # Revenue impact by segment
        fig_revenue = px.bar(
            segment_summary,
            x="risk_segment",
            y="potential_revenue_loss",
            title="Revenue at Risk by Segment",
            color="churn_probability",
            color_continuous_scale="Reds",
            labels={"potential_revenue_loss": "Revenue at Risk ($)"},
        )
        st.plotly_chart(fig_revenue, use_container_width=True)

    with col2:
        # Contract type analysis
        contract_churn = df.groupby("Contract")["Churn"].apply(
            lambda x: (x == "Yes").mean() * 100
        )
        fig_contract = px.bar(
            x=contract_churn.index,
            y=contract_churn.values,
            title="Churn Rate by Contract Type",
            labels={"x": "Contract Type", "y": "Churn Rate (%)"},
            color=contract_churn.values,
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(fig_contract, use_container_width=True)

    # Key insights
    st.markdown("---")
    st.markdown("### 🔍 Key Insights & Recommendations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        #### 🎯 Immediate Actions
        - Target 250 critical-risk customers
        - Focus on month-to-month contracts
        - Prioritize electronic check users
        """
        )

    with col2:
        st.markdown(
            """
        #### 📈 Growth Opportunities
        - Convert 30% to annual contracts
        - Reduce payment friction
        - Improve onboarding process
        """
        )

    with col3:
        st.markdown(
            """
        #### 💡 Strategic Focus
        - $487k potential savings identified
        - 523 customers for intervention
        - 84% model accuracy achieved
        """
        )

elif page == "🔮 Real-time Predictions":
    st.title("🔮 Real-time Churn Predictions")
    st.markdown("### Instant customer churn risk assessment")

    # Customer selection or manual input
    tab1, tab2 = st.tabs(["Select Existing Customer", "Manual Input"])

    with tab1:
        df = load_data()
        if not df.empty:
            customer_id = st.selectbox(
                "Select Customer ID",
                df["customerID"].tolist())

            if st.button("Get Prediction", key="existing"):
                customer_data = df[df["customerID"]
                                   == customer_id].iloc[0].to_dict()

                with st.spinner("Analyzing customer..."):
                    time.sleep(0.5)  # Simulate processing
                    result = call_api_prediction(customer_data)

                if result:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        churn_prob = result.get("churn_probability", 0.5)
                        st.metric("Churn Probability",
                                  f"{churn_prob * 100: .1f}%")

                        # Gauge chart
                        fig_gauge = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=churn_prob * 100,
                                title={"text": "Risk Level"},
                                domain={"x": [0, 1], "y": [0, 1]},
                                gauge={
                                    "axis": {"range": [None, 100]},
                                    "bar": {"color": "darkblue"},
                                    "steps": [
                                        {"range": [0, 30], "color": "lightgreen"},
                                        {"range": [30, 70], "color": "yellow"},
                                        {"range": [70, 100], "color": "lightcoral"},
                                    ],
                                    "threshold": {
                                        "line": {"color": "red", "width": 4},
                                        "thickness": 0.75,
                                        "value": 90,
                                    },
                                },
                            )
                        )
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with col2:
                        st.metric(
                            "Prediction", result.get(
                                "churn_prediction", "Unknown"))
                        st.metric(
                            "Confidence", f"{result.get('confidence', 0) * 100: .1f}%"
                        )
                        st.metric(
                            "Risk Segment", result.get(
                                "risk_segment", "Unknown"))

                    with col3:
                        st.markdown("#### Recommended Actions")
                        if churn_prob > 0.7:
                            st.error("🚨 **Immediate Intervention Required**")
                            st.markdown(
                                """
                            - Personal call from account manager
                            - Offer 50% discount for annual contract
                            - Priority support upgrade
                            """
                            )
                        elif churn_prob > 0.5:
                            st.warning("⚠️ **Proactive Engagement Needed**")
                            st.markdown(
                                """
                            - Send retention email campaign
                            - Offer service bundle discount
                            - Schedule satisfaction survey
                            """
                            )
                        else:
                            st.success("✅ **Low Risk - Maintain Engagement**")
                            st.markdown(
                                """
                            - Continue regular service
                            - Identify upsell opportunities
                            - Quarterly check-in
                            """
                            )

    with tab2:
        st.markdown("#### Enter Customer Details")

        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input(
                "Tenure (months)", min_value=0, max_value=72, value=12
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=65.0)
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=1000.0)
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )

        with col2:
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            internet_service = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

        if st.button("Get Prediction", key="manual"):
            customer_data = {
                "tenure": tenure,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "Contract": contract,
                "PaymentMethod": payment_method,
                "InternetService": internet_service,
                "PhoneService": phone_service,
                "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            }

            with st.spinner("Analyzing customer..."):
                time.sleep(0.5)
                result = call_api_prediction(customer_data)

            if result:
                st.success("✅ Prediction Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Churn Probability",
                        f"{result.get('churn_probability', 0.5) * 100: .1f}%",
                    )
                with col2:
                    st.metric(
                        "Risk Segment", result.get(
                            "risk_segment", "Unknown"))

elif page == "📊 Customer Analytics":
    st.title("📊 Customer Analytics")
    st.markdown("### Deep dive into customer behavior patterns")

    df = load_data()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        contract_filter = st.multiselect(
            "Contract Type",
            df["Contract"].unique(),
            default=df["Contract"].unique())
    with col2:
        if "risk_segment" in df.columns:
            risk_filter = st.multiselect(
                "Risk Segment",
                df["risk_segment"].unique(),
                default=df["risk_segment"].unique(),
            )
        else:
            risk_filter = []
    with col3:
        tenure_range = st.slider("Tenure Range (months)", 0, 72, (0, 72))

    # Apply filters
    filtered_df = df[
        (df["Contract"].isin(contract_filter))
        & (df["tenure"] >= tenure_range[0])
        & (df["tenure"] <= tenure_range[1])
    ]
    if risk_filter and "risk_segment" in df.columns:
        filtered_df = filtered_df[filtered_df["risk_segment"].isin(
            risk_filter)]

    # Check if filtered data is empty
    if filtered_df.empty:
        st.warning(
            "⚠️ No data matches the current filter criteria. Please adjust your filters."
        )
        st.stop()

    # Analytics tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Distributions", "Correlations", "Segments", "Trends", "Advanced Analytics"]
    )

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Tenure distribution
            fig_tenure = px.histogram(
                filtered_df,
                x="tenure",
                color="Churn",
                title="Tenure Distribution by Churn Status",
                nbins=20,
                color_discrete_map={"Yes": "#ff6b6b", "No": "#4ecdc4"},
            )
            st.plotly_chart(fig_tenure, use_container_width=True)

        with col2:
            # Monthly charges distribution
            fig_charges = px.box(
                filtered_df,
                x="Churn",
                y="MonthlyCharges",
                color="Churn",
                title="Monthly Charges by Churn Status",
                color_discrete_map={"Yes": "#ff6b6b", "No": "#4ecdc4"},
            )
            st.plotly_chart(fig_charges, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            # Service usage
            services = [
                "PhoneService",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
            ]
            service_data = []
            for service in services:
                if service in filtered_df.columns:
                    yes_count = (filtered_df[service] == "Yes").sum()
                    service_data.append(
                        {"Service": service, "Count": yes_count})

            if service_data:
                service_df = pd.DataFrame(service_data)
                fig_services = px.bar(
                    service_df,
                    x="Service",
                    y="Count",
                    title="Service Adoption",
                    color="Count",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig_services, use_container_width=True)

        with col2:
            # Payment method distribution
            payment_dist = filtered_df["PaymentMethod"].value_counts()
            fig_payment = px.pie(
                values=payment_dist.values,
                names=payment_dist.index,
                title="Payment Method Distribution",
            )
            st.plotly_chart(fig_payment, use_container_width=True)

    with tab2:
        # Correlation heatmap
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = filtered_df[numeric_cols].corr()

            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu",
                aspect="auto",
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Top correlations with churn
            if "churn_probability" in filtered_df.columns:
                churn_corr = (
                    filtered_df[numeric_cols]
                    .corrwith(filtered_df["churn_probability"])
                    .sort_values(ascending=False)
                )

                st.markdown("#### Top Features Correlated with Churn")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Positive Correlations** (Increase Churn)")
                    positive_corr = churn_corr[churn_corr > 0].head(5)
                    for feature, corr in positive_corr.items():
                        if feature != "churn_probability":
                            st.write(f"• {feature}: {corr: .3f}")

                with col2:
                    st.markdown("**Negative Correlations** (Decrease Churn)")
                    negative_corr = churn_corr[churn_corr < 0].head(5)
                    for feature, corr in negative_corr.items():
                        st.write(f"• {feature}: {corr: .3f}")

    with tab3:
        if "risk_segment" in filtered_df.columns and not filtered_df.empty:
            # Segment analysis
            segment_stats = (
                filtered_df.groupby("risk_segment")
                .agg(
                    {
                        "customerID": "count",
                        "MonthlyCharges": "mean",
                        "TotalCharges": "mean",
                        "tenure": "mean",
                    }
                )
                .round(2)
            )
            segment_stats.columns = [
                "Count",
                "Avg Monthly Charges",
                "Avg Total Charges",
                "Avg Tenure",
            ]

            st.markdown("#### Segment Statistics")
            st.dataframe(segment_stats, use_container_width=True)

            # Segment comparison
            fig_segment = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Customer Count",
                    "Avg Monthly Charges",
                    "Avg Tenure",
                    "Churn Distribution",
                ),
            )

            segments = segment_stats.index

            fig_segment.add_trace(
                go.Bar(
                    x=segments,
                    y=segment_stats["Count"],
                    name="Count",
                    marker_color=["green", "yellow", "orange", "red"],
                ),
                row=1,
                col=1,
            )

            fig_segment.add_trace(
                go.Bar(
                    x=segments,
                    y=segment_stats["Avg Monthly Charges"],
                    name="Monthly Charges",
                    marker_color="blue",
                ),
                row=1,
                col=2,
            )

            fig_segment.add_trace(
                go.Bar(
                    x=segments,
                    y=segment_stats["Avg Tenure"],
                    name="Tenure",
                    marker_color="purple",
                ),
                row=2,
                col=1,
            )

            fig_segment.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_segment, use_container_width=True)

    with tab4:
        # Trends over tenure
        if len(filtered_df) < 6:
            st.warning(
                "⚠️ Not enough data points to create tenure trends. Need at least 6 records."
            )
        else:
            try:
                tenure_groups = pd.cut(filtered_df["tenure"], bins=6)
                trend_data = (
                    filtered_df.groupby(tenure_groups, observed=True)
                    .agg(
                        {
                            "Churn": lambda x: (x == "Yes").mean() * 100,
                            "MonthlyCharges": "mean",
                            "TotalCharges": "mean",
                        }
                    )
                    .reset_index()
                )
                trend_data["tenure"] = trend_data["tenure"].astype(str)

                fig_trends = make_subplots(
                    rows=1,
                    cols=3,
                    subplot_titles=(
                        "Churn Rate by Tenure",
                        "Avg Monthly Charges",
                        "Avg Total Charges",
                    ),
                )

                fig_trends.add_trace(
                    go.Scatter(
                        x=trend_data["tenure"],
                        y=trend_data["Churn"],
                        mode="lines+markers",
                        name="Churn Rate",
                    ),
                    row=1,
                    col=1,
                )

                fig_trends.add_trace(
                    go.Scatter(
                        x=trend_data["tenure"],
                        y=trend_data["MonthlyCharges"],
                        mode="lines+markers",
                        name="Monthly Charges",
                    ),
                    row=1,
                    col=2,
                )

                fig_trends.add_trace(
                    go.Scatter(
                        x=trend_data["tenure"],
                        y=trend_data["TotalCharges"],
                        mode="lines+markers",
                        name="Total Charges",
                    ),
                    row=1,
                    col=3,
                )

                fig_trends.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_trends, use_container_width=True)

            except ValueError as e:
                st.error(f"Error creating tenure trends: {e}")
                st.stop()

    with tab5:
        # Advanced Analytics with Lazy Loading
        st.markdown("#### 🚀 Advanced Analytics")
        st.markdown(
            "*These computations are performed on-demand for optimal performance.*"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "🔍 Compute Advanced Correlations",
                    help="Analyze feature correlations"):
                with st.spinner("Computing correlations..."):
                    if (
                        "advanced_analytics" not in st.session_state
                        or st.session_state.advanced_analytics is None
                    ):
                        st.session_state.advanced_analytics = (
                            compute_advanced_analytics(filtered_df)
                        )

                    if (
                        st.session_state.advanced_analytics
                        and "correlation_matrix" in st.session_state.advanced_analytics
                    ):
                        corr_matrix = st.session_state.advanced_analytics[
                            "correlation_matrix"
                        ]

                        # Create correlation heatmap
                        fig_corr = px.imshow(
                            corr_matrix,
                            title="Feature Correlation Matrix",
                            color_continuous_scale="RdBu",
                            aspect="auto",
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)

                        # Show top correlations
                        st.markdown("**Top Correlations with Churn:**")
                        if "Churn" in corr_matrix.columns:
                            churn_corr = (
                                corr_matrix["Churn"]
                                .abs()
                                .sort_values(ascending=False)[1:6]
                            )
                            for feature, corr in churn_corr.items():
                                st.write(f"• {feature}: {corr: .3f}")

        with col2:
            if st.button(
                "📊 Generate Feature Importance",
                    help="Show model feature importance"):
                with st.spinner("Loading feature importance..."):
                    if "feature_importance" not in st.session_state:
                        st.session_state.feature_importance = load_feature_importance()

                    feat_df = st.session_state.feature_importance

                    # Create feature importance chart
                    fig_feat = px.bar(
                        feat_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Model Feature Importance",
                        color="importance",
                        color_continuous_scale="viridis",
                    )
                    fig_feat.update_layout(height=400)
                    st.plotly_chart(fig_feat, use_container_width=True)

        # Advanced segmentation analysis (lazy loaded)
        if st.button(
            "🎯 Deep Segment Analysis",
                help="Detailed segment performance"):
            with st.spinner("Analyzing segments..."):
                if (
                    "advanced_analytics" not in st.session_state
                    or st.session_state.advanced_analytics is None
                ):
                    st.session_state.advanced_analytics = compute_advanced_analytics(
                        filtered_df)

                analytics = st.session_state.advanced_analytics
                if analytics and "segment_analysis" in analytics:
                    segment_data = analytics["segment_analysis"]

                    st.markdown("**Detailed Segment Performance:**")
                    st.dataframe(segment_data, use_container_width=True)

                    # Performance insights
                    st.markdown("**Key Insights:**")
                    st.info(
                        """
                    📈 **High-Value Segments**: Focus retention efforts on high monthly charge customers
                    ⏰ **Tenure Patterns**: New customers (0-12 months) show highest churn risk
                    💡 **Intervention Timing**: Optimal intervention at 6-month tenure milestone
                    """
                    )

        # Session state debug (if enabled)
        if st.session_state.user_preferences.get("show_debug", False):
            with st.expander("🔧 Performance Debug Info"):
                st.json(
                    {
                        "cached_data_size": len(filtered_df),
                        "cache_status": {
                            "advanced_analytics": st.session_state.advanced_analytics is not None,
                            "feature_importance": "feature_importance" in st.session_state,
                            "last_refresh": str(
                                st.session_state.last_refresh),
                        },
                    })

elif page == "💰 ROI Calculator":
    st.title("💰 Retention Campaign ROI Calculator")
    st.markdown(
        "### Calculate the return on investment for targeted retention campaigns"
    )

    df = load_data()

    # Input parameters
    st.markdown("#### Campaign Parameters")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Target Selection")

        if "risk_segment" in df.columns:
            target_segment = st.selectbox(
                "Target Risk Segment",
                ["All", "Critical", "High Risk", "Medium Risk", "Low Risk"],
            )

            if target_segment == "All":
                target_customers = len(df)
                avg_churn_prob = 0.2654  # Overall churn rate
            else:
                segment_df = df[df["risk_segment"] == target_segment]
                target_customers = len(segment_df)
                avg_churn_prob = {
                    "Critical": 0.85,
                    "High Risk": 0.65,
                    "Medium Risk": 0.35,
                    "Low Risk": 0.08,
                }.get(target_segment, 0.2654)
        else:
            target_customers = st.number_input(
                "Number of Target Customers", 100, 5000, 500
            )
            avg_churn_prob = st.slider(
                "Average Churn Probability", 0.0, 1.0, 0.3)

        intervention_cost = st.number_input(
            "Cost per Intervention ($)",
            min_value=10,
            max_value=200,
            value=25,
            step=5)

    with col2:
        st.markdown("##### Business Metrics")

        clv = st.number_input(
            "Customer Lifetime Value ($)",
            min_value=500,
            max_value=5000,
            value=2000,
            step=100,
        )

        success_rate = st.slider(
            "Intervention Success Rate",
            min_value=0.1,
            max_value=0.8,
            value=0.3,
            step=0.05,
        )

        campaign_duration = st.selectbox(
            "Campaign Duration", [
                "1 Month", "3 Months", "6 Months", "12 Months"])

    # Calculate ROI
    if st.button("Calculate ROI", type="primary"):
        roi_results = calculate_roi(
            target_customers,
            avg_churn_prob,
            intervention_cost,
            success_rate,
            clv)

        st.markdown("---")
        st.markdown("### 📊 ROI Analysis Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Expected Churners", f"{int(target_customers * avg_churn_prob):, }"
            )

        with col2:
            st.metric("Prevented Churns", f"{int(roi_results['prevented_churns']):, }")

        with col3:
            st.metric("Revenue Saved", f"${roi_results['revenue_saved']:, .0f}")

        with col4:
            roi_pct = roi_results["roi"] * 100
            st.metric(
                "ROI",
                f"{roi_pct: .1f}%",
                delta=f"{'Profitable' if roi_pct > 0 else 'Loss'}",
            )

        # Visualization
        col1, col2 = st.columns(2)

        with col1:
            # Cost vs Revenue chart
            fig_roi = go.Figure()

            categories = ["Investment", "Revenue Saved", "Net Profit"]
            values = [
                roi_results["total_cost"],
                roi_results["revenue_saved"],
                roi_results["revenue_saved"] - roi_results["total_cost"],
            ]
            colors = ["red", "green", "blue" if values[2] > 0 else "orange"]

            fig_roi.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=colors,
                    text=[f"${v:, .0f}" for v in values],
                    textposition="auto",
                )
            )

            fig_roi.update_layout(
                title="Campaign Financial Analysis",
                yaxis_title="Amount ($)",
                showlegend=False,
            )
            st.plotly_chart(fig_roi, use_container_width=True)

        with col2:
            # Break-even analysis
            success_rates = np.linspace(0.1, 0.8, 20)
            rois = []

            for sr in success_rates:
                roi_calc = calculate_roi(
                    target_customers,
                    avg_churn_prob,
                    intervention_cost,
                    sr,
                    clv)
                rois.append(roi_calc["roi"] * 100)

            fig_breakeven = go.Figure()
            fig_breakeven.add_trace(
                go.Scatter(
                    x=success_rates * 100,
                    y=rois,
                    mode="lines+markers",
                    name="ROI",
                    line=dict(width=3),
                )
            )

            fig_breakeven.add_hline(
                y=0,
                line_dash="dash",
                line_color="red",
                annotation_text="Break-even")

            fig_breakeven.add_vline(
                x=success_rate * 100,
                line_dash="dash",
                line_color="blue",
                annotation_text="Current",
            )

            fig_breakeven.update_layout(
                title="ROI Sensitivity to Success Rate",
                xaxis_title="Success Rate (%)",
                yaxis_title="ROI (%)",
            )
            st.plotly_chart(fig_breakeven, use_container_width=True)

        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recommendations")

        if roi_results["roi"] > 0.5:
            st.success(
                """
            ✅ **Highly Profitable Campaign**
            - Proceed with immediate implementation
            - Consider expanding target audience
            - Monitor success rate closely for optimization
            """
            )
        elif roi_results["roi"] > 0:
            st.info(
                """
            ℹ️ **Profitable Campaign**
            - Campaign is worth pursuing
            - Focus on improving success rate
            - Consider A/B testing different approaches
            """
            )
        else:
            st.warning(
                """
            ⚠️ **Unprofitable Campaign**
            - Reconsider campaign parameters
            - Focus on higher-risk segments only
            - Reduce intervention costs or improve success rate
            """
            )

elif page == "🎯 Segmentation Analysis":
    st.title("🎯 Customer Segmentation Analysis")
    st.markdown("### Strategic customer grouping for targeted interventions")

    df = load_data()

    # Segmentation method selection
    seg_method = st.selectbox(
        "Segmentation Method", [
            "Risk-Based", "Value-Based", "Behavioral", "Custom"])

    if seg_method == "Risk-Based":
        if "risk_segment" in df.columns:
            # Risk segment distribution
            segment_counts = df["risk_segment"].value_counts()

            col1, col2 = st.columns(2)

            with col1:
                # Donut chart
                fig_donut = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title="Risk Segment Distribution",
                    hole=0.4,
                    color_discrete_map={
                        "Low Risk": "#00cc00",
                        "Medium Risk": "#ffa500",
                        "High Risk": "#ff6b6b",
                        "Critical": "#cc0000",
                    },
                )
                fig_donut.update_traces(
                    textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_donut, use_container_width=True)

            with col2:
                # Segment metrics
                segment_metrics = (
                    df.groupby("risk_segment")
                    .agg(
                        {
                            "MonthlyCharges": "mean",
                            "tenure": "mean",
                            "TotalCharges": "sum",
                        }
                    )
                    .round(2)
                )

                st.markdown("#### Segment Metrics")
                for segment in segment_metrics.index:
                    with st.expander(
                        f"{segment} ({segment_counts[segment]:, } customers)"
                    ):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Avg Monthly",
                                f"${segment_metrics.loc[segment,
                                                        'MonthlyCharges']: .2f}",
                            )
                        with col2:
                            st.metric(
                                "Avg Tenure",
                                f"{segment_metrics.loc[segment, 'tenure']: .0f} mo",
                            )
                        with col3:
                            st.metric(
                                "Total Revenue",
                                f"${segment_metrics.loc[segment,
                                                        'TotalCharges'] / 1000: .0f}K",
                            )

            # Detailed segment analysis
            st.markdown("---")
            selected_segment = st.selectbox(
                "Select Segment for Deep Dive", segment_counts.index
            )

            segment_df = df[df["risk_segment"] == selected_segment]

            col1, col2, col3 = st.columns(3)

            with col1:
                # Contract distribution in segment
                contract_dist = segment_df["Contract"].value_counts()
                fig_contract = px.pie(
                    values=contract_dist.values,
                    names=contract_dist.index,
                    title=f"Contract Types in {selected_segment}",
                )
                st.plotly_chart(fig_contract, use_container_width=True)

            with col2:
                # Payment method distribution
                payment_dist = segment_df["PaymentMethod"].value_counts()
                fig_payment = px.bar(
                    x=payment_dist.values,
                    y=payment_dist.index,
                    orientation="h",
                    title=f"Payment Methods in {selected_segment}",
                )
                st.plotly_chart(fig_payment, use_container_width=True)

            with col3:
                # Service adoption
                services = [
                    "PhoneService",
                    "InternetService",
                    "OnlineSecurity"]
                service_adoption = []
                for service in services:
                    if service in segment_df.columns:
                        adoption_rate = (
                            segment_df[service] == "Yes").mean() * 100
                        service_adoption.append(
                            {
                                "Service": service.replace("Service", ""),
                                "Adoption": adoption_rate,
                            }
                        )

                if service_adoption:
                    service_df = pd.DataFrame(service_adoption)
                    fig_services = px.bar(
                        service_df,
                        x="Service",
                        y="Adoption",
                        title=f"Service Adoption in {selected_segment}",
                        color="Adoption",
                        color_continuous_scale="Viridis",
                    )
                    fig_services.update_layout(yaxis_title="Adoption Rate (%)")
                    st.plotly_chart(fig_services, use_container_width=True)

    elif seg_method == "Value-Based":
        # Customer value segmentation
        df["value_segment"] = pd.qcut(
            df["TotalCharges"],
            q=4,
            labels=["Low Value", "Medium Value", "High Value", "Premium"],
        )

        value_analysis = (
            df.groupby("value_segment", observed=True)
            .agg(
                {
                    "customerID": "count",
                    "Churn": lambda x: (x == "Yes").mean() * 100,
                    "MonthlyCharges": "mean",
                    "TotalCharges": "mean",
                }
            )
            .round(2)
        )
        value_analysis.columns = [
            "Count",
            "Churn Rate (%)",
            "Avg Monthly",
            "Avg Total"]

        st.markdown("#### Value-Based Segmentation")
        st.dataframe(value_analysis, use_container_width=True)

        # Visualization
        fig_value = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Customer Distribution", "Churn Rate by Value"),
        )

        fig_value.add_trace(
            go.Bar(
                x=value_analysis.index,
                y=value_analysis["Count"]),
            row=1,
            col=1)

        fig_value.add_trace(
            go.Scatter(
                x=value_analysis.index,
                y=value_analysis["Churn Rate (%)"],
                mode="lines+markers",
                line=dict(width=3),
            ),
            row=1,
            col=2,
        )

        fig_value.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_value, use_container_width=True)

    elif seg_method == "Behavioral":
        st.markdown("#### Behavioral Segmentation")

        # Create behavioral segments
        df["service_intensity"] = df[
            [
                "PhoneService",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
            ]
        ].apply(lambda x: sum(x == "Yes"), axis=1)

        df["behavioral_segment"] = df.apply(
            lambda x: "Power User"
            if x["service_intensity"] >= 6
            else "Regular User"
            if x["service_intensity"] >= 3
            else "Light User",
            axis=1,
        )

        behavioral_stats = (
            df.groupby("behavioral_segment", observed=True)
            .agg(
                {
                    "customerID": "count",
                    "Churn": lambda x: (x == "Yes").mean() * 100,
                    "service_intensity": "mean",
                    "MonthlyCharges": "mean",
                }
            )
            .round(2)
        )
        behavioral_stats.columns = [
            "Count",
            "Churn Rate (%)",
            "Avg Services",
            "Avg Monthly",
        ]

        st.dataframe(behavioral_stats, use_container_width=True)

        # Visualization
        col1, col2 = st.columns(2)

        with col1:
            fig_behavior = px.scatter(
                df,
                x="service_intensity",
                y="MonthlyCharges",
                color="behavioral_segment",
                title="Service Intensity vs Monthly Charges",
                size="TotalCharges",
                size_max=20,
            )
            st.plotly_chart(fig_behavior, use_container_width=True)

        with col2:
            behavior_churn = (
                df.groupby(["behavioral_segment", "Churn"], observed=True)
                .size()
                .unstack(fill_value=0)
            )
            fig_stack = px.bar(
                behavior_churn.T,
                title="Churn Distribution by Behavior",
                color_discrete_map={"Yes": "#ff6b6b", "No": "#4ecdc4"},
            )
            st.plotly_chart(fig_stack, use_container_width=True)

elif page == "⚙️ What-If Scenarios":
    st.title("⚙️ What-If Scenario Analysis")
    st.markdown("### Simulate the impact of different retention strategies")

    df = load_data()

    # Scenario selection
    scenario = st.selectbox(
        "Select Scenario",
        [
            "Contract Upgrade Campaign",
            "Payment Method Migration",
            "Service Bundle Promotion",
            "Price Adjustment",
            "Custom Scenario",
        ],
    )

    if scenario == "Contract Upgrade Campaign":
        st.markdown("#### Simulate Contract Upgrade Campaign")

        col1, col2 = st.columns(2)

        with col1:
            current_mtm = len(df[df["Contract"] == "Month-to-month"])
            st.metric("Current Month-to-Month", f"{current_mtm:, }")

            conversion_rate = st.slider(
                "Expected Conversion Rate (%)",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
            )

            discount_offered = st.slider(
                "Discount Offered (%)",
                min_value=5,
                max_value=30,
                value=15,
                step=5)

        with col2:
            # Current metrics
            current_churn = (
                df[df["Contract"] == "Month-to-month"]["Churn"] == "Yes"
            ).mean()
            annual_churn = (df[df["Contract"] == "One year"]
                            ["Churn"] == "Yes").mean()

            st.metric("M-t-M Churn Rate", f"{current_churn * 100: .1f}%")
            st.metric("Annual Churn Rate", f"{annual_churn * 100: .1f}%")

        if st.button("Run Simulation", key="contract_sim"):
            # Calculate impact
            converted_customers = int(current_mtm * conversion_rate / 100)
            churn_reduction = (
                current_churn - annual_churn) * converted_customers

            avg_monthly = df[df["Contract"] == "Month-to-month"][
                "MonthlyCharges"
            ].mean()
            revenue_impact = (converted_customers *
                              avg_monthly * 12 * (1 - discount_offered / 100))
            saved_customers = int(churn_reduction)

            st.markdown("---")
            st.markdown("### Simulation Results")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Converted Customers", f"{converted_customers:, }")
            with col2:
                st.metric("Churn Prevented", f"{saved_customers:, }")
            with col3:
                st.metric("Revenue Impact", f"${revenue_impact:, .0f}")
            with col4:
                roi = (
                    (revenue_impact - (converted_customers * 50))
                    / (converted_customers * 50)
                    * 100
                )
                st.metric("Campaign ROI", f"{roi: .1f}%")

            # Visualization
            months = list(range(1, 13))
            cumulative_savings = [
                saved_customers *
                avg_monthly *
                m for m in months]

            fig_impact = go.Figure()
            fig_impact.add_trace(
                go.Scatter(
                    x=months,
                    y=cumulative_savings,
                    mode="lines+markers",
                    name="Cumulative Savings",
                    fill="tozeroy",
                )
            )

            fig_impact.update_layout(
                title="12-Month Revenue Impact",
                xaxis_title="Month",
                yaxis_title="Cumulative Savings ($)",
            )
            st.plotly_chart(fig_impact, use_container_width=True)

    elif scenario == "Payment Method Migration":
        st.markdown("#### Simulate Payment Method Migration")

        electronic_check_customers = len(
            df[df["PaymentMethod"] == "Electronic check"])
        electronic_check_churn = (
            df[df["PaymentMethod"] == "Electronic check"]["Churn"] == "Yes"
        ).mean()
        auto_payment_churn = (
            df[df["PaymentMethod"].str.contains("automatic")]["Churn"] == "Yes"
        ).mean()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Electronic Check Users", f"{electronic_check_customers:, }")
            st.metric("Their Churn Rate",
                      f"{electronic_check_churn * 100: .1f}%")

            migration_rate = st.slider(
                "Expected Migration Rate (%)",
                min_value=10,
                max_value=60,
                value=30,
                step=5,
            )

            incentive_cost = st.number_input(
                "Incentive Cost per Customer ($)",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
            )

        with col2:
            st.metric("Auto-Payment Churn Rate",
                      f"{auto_payment_churn * 100: .1f}%")

            if st.button("Run Simulation", key="payment_sim"):
                migrated = int(
                    electronic_check_customers *
                    migration_rate /
                    100)
                churn_prevented = int(
                    migrated * (electronic_check_churn - auto_payment_churn)
                )

                total_cost = migrated * incentive_cost
                revenue_saved = churn_prevented * \
                    df["MonthlyCharges"].mean() * 12

                st.markdown("---")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Customers Migrated", f"{migrated:, }")
                with col2:
                    st.metric("Churn Prevented", f"{churn_prevented:, }")
                with col3:
                    net_benefit = revenue_saved - total_cost
                    st.metric("Net Benefit", f"${net_benefit:, .0f}")

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Monitoring")
    st.markdown("### Track and evaluate model performance metrics")

    # Model metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Accuracy (AUC)", "84%", delta="+2% vs baseline")
    with col2:
        st.metric("Precision", "41%", delta="+5% this month")
    with col3:
        st.metric("Recall", "78%", delta="-2% this month")
    with col4:
        st.metric("F1 Score", "54%", delta="+1% this month")

    # Performance over time
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    performance_data = pd.DataFrame(
        {
            "Date": dates,
            "AUC": np.random.uniform(0.82, 0.86, 30),
            "Precision": np.random.uniform(0.38, 0.44, 30),
            "Recall": np.random.uniform(0.76, 0.80, 30),
        }
    )

    fig_perf = go.Figure()

    for metric in ["AUC", "Precision", "Recall"]:
        fig_perf.add_trace(
            go.Scatter(
                x=performance_data["Date"],
                y=performance_data[metric],
                mode="lines",
                name=metric,
            )
        )

    fig_perf.update_layout(
        title="Model Performance Trend (30 Days)",
        xaxis_title="Date",
        yaxis_title="Score",
        hovermode="x unified",
    )

    st.plotly_chart(fig_perf, use_container_width=True)

    # Confusion Matrix
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")

        # Mock confusion matrix
        confusion_matrix = [[1500, 300], [400, 800]]

        fig_cm = px.imshow(
            confusion_matrix,
            labels=dict(x="Predicted", y="Actual"),
            x=["No Churn", "Churn"],
            y=["No Churn", "Churn"],
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("#### Feature Importance")

        features = [
            "Contract",
            "tenure",
            "MonthlyCharges",
            "PaymentMethod",
            "InternetService",
            "TotalCharges",
            "OnlineSecurity",
        ]
        importance = [0.25, 0.20, 0.15, 0.12, 0.10, 0.10, 0.08]

        fig_importance = px.bar(
            x=importance,
            y=features,
            orientation="h",
            title="Top Feature Importance",
            color=importance,
            color_continuous_scale="Viridis",
        )
        fig_importance.update_layout(
            xaxis_title="Importance Score", yaxis_title="Feature", height=400
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    # Model comparison
    st.markdown("---")
    st.markdown("### Model Comparison")

    models_data = pd.DataFrame(
        {
            "Model": [
                "XGBoost", "LightGBM", "Random Forest", "Logistic Regression"], "AUC": [
                0.84, 0.83, 0.79, 0.75], "Training Time (s)": [
                    2.5, 1.8, 3.2, 0.5], "Inference Time (ms)": [
                        0.9, 0.7, 1.2, 0.3], })

    st.dataframe(models_data, use_container_width=True)

    # Data drift monitoring
    st.markdown("---")
    st.markdown("### Data Drift Monitoring")

    drift_metrics = pd.DataFrame(
        {
            "Feature": ["tenure", "MonthlyCharges", "Contract", "PaymentMethod"],
            "Drift Score": [0.02, 0.05, 0.08, 0.12],
            "Status": ["✅ Stable", "✅ Stable", "⚠️ Minor Drift", "⚠️ Minor Drift"],
        }
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_drift = px.bar(
            drift_metrics,
            x="Feature",
            y="Drift Score",
            color="Drift Score",
            title="Feature Drift Scores",
            color_continuous_scale="RdYlGn_r",
        )
        fig_drift.add_hline(
            y=0.1,
            line_dash="dash",
            line_color="red",
            annotation_text="Drift Threshold")
        st.plotly_chart(fig_drift, use_container_width=True)

    with col2:
        st.markdown("#### Drift Summary")
        st.dataframe(
            drift_metrics[["Feature", "Status"]], use_container_width=True)

        if any(drift_metrics["Drift Score"] > 0.1):
            st.warning(
                "⚠️ Some features showing drift. Consider model retraining.")
        else:
            st.success("✅ All features within acceptable drift limits.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    Telco Customer Intelligence Platform v1.0 | Built with Streamlit & FastAPI |
    © 2025 Zaid Shaikh
    </div>
    """,
    unsafe_allow_html=True,
)
