"""
Pydantic schemas for API request/response validation

Location: src/api/schemas/models.py
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class ContractType(str, Enum):
    """Contract type enumeration."""

    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"


class InternetServiceType(str, Enum):
    """Internet service type enumeration."""

    DSL = "DSL"
    FIBER = "Fiber optic"
    NO = "No"


class PaymentMethodType(str, Enum):
    """Payment method enumeration."""

    ELECTRONIC_CHECK = "Electronic check"
    MAILED_CHECK = "Mailed check"
    BANK_TRANSFER = "Bank transfer (automatic)"
    CREDIT_CARD = "Credit card (automatic)"


class CustomerFeatures(BaseModel):
    """Input features for churn prediction."""

    # Demographics
    gender: str = Field(..., description="Customer gender (Male/Female)")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Is senior citizen (0/1)")
    Partner: str = Field(..., description="Has partner (Yes/No)")
    Dependents: str = Field(..., description="Has dependents (Yes/No)")

    # Account information
    tenure: int = Field(..., ge=0, le=72, description="Number of months with company")
    Contract: str = Field(..., description="Contract type")  # Changed back to str
    PaperlessBilling: str = Field(..., description="Paperless billing (Yes/No)")
    PaymentMethod: str = Field(..., description="Payment method")  # Changed back to str

    # Services
    PhoneService: str = Field(..., description="Phone service (Yes/No)")
    MultipleLines: str = Field(
        "No", description="Multiple lines (Yes/No/No phone service)"
    )
    InternetService: str = Field(
        ..., description="Internet service type"
    )  # Changed back to str
    OnlineSecurity: str = Field(
        "No", description="Online security (Yes/No/No internet service)"
    )
    OnlineBackup: str = Field(
        "No", description="Online backup (Yes/No/No internet service)"
    )
    DeviceProtection: str = Field(
        "No", description="Device protection (Yes/No/No internet service)"
    )
    TechSupport: str = Field(
        "No", description="Tech support (Yes/No/No internet service)"
    )
    StreamingTV: str = Field(
        "No", description="Streaming TV (Yes/No/No internet service)"
    )
    StreamingMovies: str = Field(
        "No", description="Streaming movies (Yes/No/No internet service)"
    )

    # Charges
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in USD")
    TotalCharges: float = Field(..., ge=0, description="Total charges in USD")

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.20,
            }
        }

    @validator("TotalCharges")
    def validate_total_charges(cls, v, values):
        """Ensure TotalCharges is reasonable given tenure and MonthlyCharges."""
        if "tenure" in values and "MonthlyCharges" in values:
            expected_max = values["tenure"] * values["MonthlyCharges"]
            if v > expected_max * 1.5:  # Allow some variance
                raise ValueError(
                    "TotalCharges seems too high given tenure and MonthlyCharges"
                )
        return v

    @validator("Contract")
    def validate_contract(cls, v):
        """Validate contract type."""
        valid_contracts = ["Month-to-month", "One year", "Two year"]
        if v not in valid_contracts:
            raise ValueError(f"Contract must be one of {valid_contracts}")
        return v

    @validator("PaymentMethod")
    def validate_payment_method(cls, v):
        """Validate payment method."""
        valid_methods = [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
        if v not in valid_methods:
            raise ValueError(f"PaymentMethod must be one of {valid_methods}")
        return v

    @validator("InternetService")
    def validate_internet_service(cls, v):
        """Validate internet service."""
        valid_services = ["DSL", "Fiber optic", "No"]
        if v not in valid_services:
            raise ValueError(f"InternetService must be one of {valid_services}")
        return v


class ChurnPredictionResponse(BaseModel):
    """Response model for churn prediction."""

    customer_id: Optional[str] = Field(None, description="Customer ID if provided")
    churn_probability: float = Field(
        ..., ge=0, le=1, description="Probability of churn (0-1)"
    )
    churn_prediction: str = Field(..., description="Binary prediction (Yes/No)")
    risk_segment: str = Field(
        ..., description="Risk category (Low/Medium/High/Critical)"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Model confidence in prediction"
    )
    monthly_value_at_risk: float = Field(..., description="Monthly revenue at risk")
    recommended_action: str = Field(..., description="Recommended retention action")

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "customer_id": "7590-VHVEG",
                "churn_probability": 0.73,
                "churn_prediction": "Yes",
                "risk_segment": "High Risk",
                "confidence": 0.85,
                "monthly_value_at_risk": 70.35,
                "recommended_action": "Immediate: Offer 50% discount for 1-year contract upgrade",
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    customers: List[CustomerFeatures] = Field(
        ..., description="List of customers to predict"
    )
    include_recommendations: bool = Field(
        True, description="Include action recommendations"
    )

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "customers": [CustomerFeatures.Config.schema_extra["example"]],
                "include_recommendations": True,
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[ChurnPredictionResponse]
    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    processing_time_seconds: float

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "predictions": [ChurnPredictionResponse.Config.schema_extra["example"]],
                "summary": {
                    "total_customers": 100,
                    "high_risk_count": 25,
                    "total_value_at_risk": 5234.50,
                    "average_churn_probability": 0.35,
                },
                "processing_time_seconds": 0.234,
            }
        }


class ModelMetricsResponse(BaseModel):
    """Response model for model metrics."""

    model_version: str
    training_date: str
    performance_metrics: Dict[str, float]
    feature_importance: List[Dict[str, Any]]
    data_statistics: Dict[str, Any]

    class Config:
        """Pydantic config."""

        schema_extra = {
            "example": {
                "model_version": "1.0.0",
                "training_date": "2025-01-15T10:30:00",
                "performance_metrics": {
                    "auc_roc": 0.83,
                    "accuracy": 0.79,
                    "precision": 0.65,
                    "recall": 0.54,
                },
                "feature_importance": [
                    {"feature": "Contract", "importance": 0.234},
                    {"feature": "tenure", "importance": 0.189},
                ],
                "data_statistics": {
                    "training_samples": 4930,
                    "test_samples": 2113,
                    "features_used": 31,
                },
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    status: str
    timestamp: datetime
    model_loaded: bool
    version: str
    uptime_seconds: float


class CustomerSegmentRequest(BaseModel):
    """Request model for customer segmentation."""

    min_churn_probability: float = Field(0.0, ge=0, le=1)
    max_churn_probability: float = Field(1.0, ge=0, le=1)
    limit: int = Field(100, ge=1, le=1000)

    @validator("max_churn_probability")
    def validate_probability_range(cls, v, values):
        """Validate probability range."""
        if "min_churn_probability" in values and v < values["min_churn_probability"]:
            raise ValueError(
                "max_churn_probability must be greater than min_churn_probability"
            )
        return v


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    path: Optional[str] = None
