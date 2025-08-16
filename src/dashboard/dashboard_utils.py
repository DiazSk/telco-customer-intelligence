"""
Dashboard Utility Functions
Reusable components and helper functions for the Streamlit dashboard
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Color schemes


RISK_COLORS = {
    "Low Risk": "#00cc00",
    "Medium Risk": "#ffa500",
    "High Risk": "#ff6b6b",
    "Critical": "#cc0000",
}


CHURN_COLORS = {"Yes": "#ff6b6b", "No": "#4ecdc4"}


def format_currency(value: float) -> str:
    """Format value as currency"""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.0f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value * 100:.1f}%"


def create_gauge_chart(value: float, title: str = "Risk Level") -> go.Figure:
    """Create a gauge chart for risk visualization"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value * 100,
            title={"text": title},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 50], "color": "yellow"},
                    {"range": [50, 70], "color": "orange"},
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
    fig.update_layout(height=300)
    return fig


def create_risk_distribution_pie(df: pd.DataFrame) -> go.Figure:
    """Create risk distribution pie chart"""
    if "risk_segment" not in df.columns:
        return None

    risk_dist = df["risk_segment"].value_counts()

    fig = px.pie(
        values=risk_dist.values,
        names=risk_dist.index,
        title="Customer Risk Distribution",
        color_discrete_map=RISK_COLORS,
        hole=0.4,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>"
        + "Count: %{value}<br>"
        + "Percentage: %{percent}<br>"
        + "<extra></extra>",
    )
    return fig


def create_trend_chart(
    df: pd.DataFrame, date_col: str, value_col: str, title: str = "Trend Analysis"
) -> go.Figure:
    """Create a trend line chart"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[value_col],
            mode="lines+markers",
            name=value_col,
            line=dict(width=3, color="#3498db"),
            marker=dict(size=8),
        )
    )

    # Add trend line
    z = np.polyfit(range(len(df)), df[value_col], 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=p(range(len(df))),
            mode="lines",
            name="Trend",
            line=dict(dash="dash", color="red"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=date_col,
        yaxis_title=value_col,
        hovermode="x unified",
        showlegend=True,
    )

    return fig


def calculate_business_metrics(df: pd.DataFrame) -> Dict:
    """Calculate key business metrics"""
    metrics = {}

    # Basic metrics
    metrics["total_customers"] = len(df)
    metrics["churn_rate"] = (df["Churn"] == "Yes").mean()
    metrics["retention_rate"] = 1 - metrics["churn_rate"]

    # Revenue metrics
    metrics["monthly_revenue"] = df["MonthlyCharges"].sum()
    metrics["avg_monthly_charges"] = df["MonthlyCharges"].mean()
    metrics["total_revenue"] = df["TotalCharges"].sum()

    # Risk metrics
    if "risk_segment" in df.columns:
        metrics["high_risk_count"] = len(
            df[df["risk_segment"].isin(["High Risk", "Critical"])]
        )
        metrics["high_risk_percentage"] = (
            metrics["high_risk_count"] / metrics["total_customers"]
        )

    # Churn impact
    churned_df = df[df["Churn"] == "Yes"]
    metrics["churned_customers"] = len(churned_df)
    metrics["revenue_lost"] = churned_df["MonthlyCharges"].sum()
    metrics["annual_revenue_lost"] = metrics["revenue_lost"] * 12

    return metrics


def create_cohort_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Create cohort analysis table"""
    # Create tenure cohorts
    df["tenure_cohort"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"],
    )

    # Calculate metrics by cohort
    cohort_analysis = (
        df.groupby("tenure_cohort")
        .agg(
            {
                "customerID": "count",
                "Churn": lambda x: (x == "Yes").mean(),
                "MonthlyCharges": "mean",
                "TotalCharges": "mean",
            }
        )
        .round(2)
    )

    cohort_analysis.columns = ["Customers", "Churn Rate", "Avg Monthly", "Avg Total"]

    return cohort_analysis


def create_segment_comparison(df: pd.DataFrame) -> go.Figure:
    """Create segment comparison chart"""
    if "risk_segment" not in df.columns:
        return None

    segment_stats = (
        df.groupby("risk_segment")
        .agg({"MonthlyCharges": "mean", "tenure": "mean", "TotalCharges": "mean"})
        .round(2)
    )

    fig = go.Figure()

    # Add bars for each metric
    metrics = ["MonthlyCharges", "tenure", "TotalCharges"]
    for metric in metrics:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=segment_stats.index,
                y=segment_stats[metric],
                text=segment_stats[metric].round(1),
                textposition="auto",
            )
        )

    fig.update_layout(
        title="Segment Comparison",
        xaxis_title="Risk Segment",
        yaxis_title="Value",
        barmode="group",
        hovermode="x unified",
    )

    return fig


def generate_recommendations(
    risk_segment: str, churn_probability: float, customer_data: Dict
) -> List[str]:
    """Generate personalized recommendations based on customer profile"""
    recommendations = []

    if risk_segment == "Critical" or churn_probability > 0.7:
        recommendations.extend(
            [
                "🚨 **Immediate intervention required**",
                "• Schedule personal call from account manager within 24 hours",
                "• Offer 50% discount for 2-year contract upgrade",
                "• Provide dedicated support channel access",
                "• Consider service credit or loyalty bonus",
            ]
        )
    elif risk_segment == "High Risk" or churn_probability > 0.5:
        recommendations.extend(
            [
                "⚠️ **Proactive engagement needed**",
                "• Send personalized retention email campaign",
                "• Offer 30% discount for annual contract",
                "• Bundle services with 20% discount",
                "• Schedule satisfaction survey with incentive",
            ]
        )
    elif risk_segment == "Medium Risk" or churn_probability > 0.3:
        recommendations.extend(
            [
                "📊 **Monitor and engage**",
                "• Include in monthly newsletter with special offers",
                "• Offer free service upgrade trial",
                "• Send product education materials",
                "• Add to loyalty program if not enrolled",
            ]
        )
    else:
        recommendations.extend(
            [
                "✅ **Maintain and upsell**",
                "• Continue regular service",
                "• Identify cross-sell opportunities",
                "• Send thank you message with referral incentive",
                "• Consider for beta testing new features",
            ]
        )

    # Add specific recommendations based on customer attributes
    if customer_data.get("Contract") == "Month-to-month":
        recommendations.append("• Priority: Convert to annual contract")

    if customer_data.get("PaymentMethod") == "Electronic check":
        recommendations.append("• Encourage automatic payment setup")

    if customer_data.get("tenure", 0) < 6:
        recommendations.append("• Enhance onboarding experience")

    return recommendations


def calculate_intervention_roi(
    num_customers: int,
    avg_churn_prob: float,
    intervention_cost: float,
    success_rate: float,
    customer_lifetime_value: float,
) -> Dict:
    """Calculate ROI for intervention campaigns"""

    # Expected outcomes
    expected_churners = num_customers * avg_churn_prob
    prevented_churns = expected_churners * success_rate

    # Financial impact
    total_cost = num_customers * intervention_cost
    revenue_saved = prevented_churns * customer_lifetime_value
    net_benefit = revenue_saved - total_cost
    roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0

    return {
        "target_customers": num_customers,
        "expected_churners": int(expected_churners),
        "prevented_churns": int(prevented_churns),
        "total_cost": total_cost,
        "revenue_saved": revenue_saved,
        "net_benefit": net_benefit,
        "roi_percentage": roi,
        "cost_per_save": total_cost / prevented_churns if prevented_churns > 0 else 0,
    }


def create_feature_importance_chart(importance_data: pd.DataFrame) -> go.Figure:
    """Create feature importance visualization"""
    fig = px.bar(
        importance_data.head(10),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Feature Importance",
        color="importance",
        color_continuous_scale="Viridis",
        labels={"importance": "Importance Score", "feature": "Feature"},
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Importance Score",
        yaxis_title="Feature",
    )

    return fig


def simulate_campaign_impact(
    df: pd.DataFrame, campaign_type: str, parameters: Dict
) -> Dict:
    """Simulate the impact of different campaign types"""

    results = {}

    if campaign_type == "contract_upgrade":
        # Simulate contract upgrade campaign
        mtm_customers = df[df["Contract"] == "Month-to-month"]
        target_size = len(mtm_customers)
        conversion_rate = parameters.get("conversion_rate", 0.2)

        converted = int(target_size * conversion_rate)
        mtm_churn = (mtm_customers["Churn"] == "Yes").mean()
        annual_churn = (df[df["Contract"] == "One year"]["Churn"] == "Yes").mean()

        churn_prevented = int(converted * (mtm_churn - annual_churn))
        revenue_impact = churn_prevented * df["MonthlyCharges"].mean() * 12

        results = {
            "target_size": target_size,
            "converted": converted,
            "churn_prevented": churn_prevented,
            "revenue_impact": revenue_impact,
        }

    elif campaign_type == "payment_migration":
        # Simulate payment method migration
        ec_customers = df[df["PaymentMethod"] == "Electronic check"]
        target_size = len(ec_customers)
        migration_rate = parameters.get("migration_rate", 0.3)

        migrated = int(target_size * migration_rate)
        ec_churn = (ec_customers["Churn"] == "Yes").mean()
        auto_churn = (
            df[df["PaymentMethod"].str.contains("automatic")]["Churn"] == "Yes"
        ).mean()

        churn_prevented = int(migrated * (ec_churn - auto_churn))
        revenue_impact = churn_prevented * df["MonthlyCharges"].mean() * 12

        results = {
            "target_size": target_size,
            "migrated": migrated,
            "churn_prevented": churn_prevented,
            "revenue_impact": revenue_impact,
        }

    return results


def export_dashboard_data(df: pd.DataFrame, metrics: Dict) -> bytes:
    """Export dashboard data to Excel"""
    from io import BytesIO

    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Write main dataframe
        df.to_excel(writer, sheet_name="Customer Data", index=False)

        # Write metrics
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

        # Write segment summary if available
        if "risk_segment" in df.columns:
            segment_summary = df.groupby("risk_segment").agg(
                {
                    "customerID": "count",
                    "MonthlyCharges": "mean",
                    "Churn": lambda x: (x == "Yes").mean(),
                }
            )
            segment_summary.to_excel(writer, sheet_name="Segment Summary")

    output.seek(0)
    return output.getvalue()


# Cache decorators for expensive operations
@st.cache_data(ttl=600)
def load_and_process_data(filepath: str) -> pd.DataFrame:
    """Load and process data with caching"""
    df = pd.read_csv(filepath)
    # Add any additional processing
    return df


@st.cache_data(ttl=300)
def compute_expensive_metrics(df: pd.DataFrame) -> Dict:
    """Compute expensive metrics with caching"""
    return calculate_business_metrics(df)
