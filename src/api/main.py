"""
FastAPI application for Telco Churn Prediction.

Location: src/api/main.py
Run with: uvicorn src.api.main:app --port 8000
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path for local imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Local imports after path setup
from src.api.schemas.models import (BatchPredictionRequest,  # noqa: E402
                                    BatchPredictionResponse,
                                    ChurnPredictionResponse, CustomerFeatures,
                                    HealthCheckResponse, ModelMetricsResponse)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
MODEL_ARTIFACTS = None
APP_START_TIME = datetime.now()
FEATURE_CACHE: Dict[str, Any] = {}
FEATURE_COLUMNS_INDICES: Dict[str, int] = {}
NUMPY_FEATURE_TEMPLATE = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown."""
    # Startup
    global MODEL_ARTIFACTS, APP_START_TIME
    APP_START_TIME = datetime.now()

    logger.info("Starting Telco Churn Prediction API...")

    try:
        import joblib
        import numpy as np

        model_path = "models/saved/churn_model_artifacts.pkl"
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}...")
            MODEL_ARTIFACTS = joblib.load(model_path)

            # Pre-compute performance optimizations
            global FEATURE_COLUMNS_INDICES, NUMPY_FEATURE_TEMPLATE
            FEATURE_COLUMNS_INDICES = {}
            for i, col in enumerate(MODEL_ARTIFACTS["feature_columns"]):
                FEATURE_COLUMNS_INDICES[col] = i
            NUMPY_FEATURE_TEMPLATE = np.zeros(
                (1, len(MODEL_ARTIFACTS["feature_columns"]))
            )

            # Warm up the model with a dummy prediction
            warmup_features = np.zeros((1, len(MODEL_ARTIFACTS["feature_columns"])))
            if MODEL_ARTIFACTS.get("scaler"):
                warmup_features = MODEL_ARTIFACTS["scaler"].transform(warmup_features)
            _ = MODEL_ARTIFACTS["model"].predict_proba(warmup_features)

            logger.info("Model loaded and optimized successfully!")
            logger.info(f"   Model type: {type(MODEL_ARTIFACTS['model'])}")
            logger.info(f"   Model AUC: {MODEL_ARTIFACTS['model_metrics']['auc']:.3f}")
            logger.info(f"   Features: {len(MODEL_ARTIFACTS['feature_columns'])}")
        else:
            logger.error(f"Model file not found at {model_path}")
            logger.info("Please run: python scripts/save_model.py")
            MODEL_ARTIFACTS = None
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        MODEL_ARTIFACTS = None

    yield  # This separates startup from shutdown

    # Shutdown
    logger.info("Shutting down API...")


# Initialize FastAPI app
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Production-ready API for predicting customer churn in telecommunications",
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "Zaid Shaikh",
        "email": "zaid07sk@gmail.com",
    },
    license_info={
        "name": "MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Helper Functions
# ============================================================================


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def prepare_features(customer: CustomerFeatures):
    """Optimized feature preparation using numpy arrays instead of pandas."""
    if MODEL_ARTIFACTS is None or NUMPY_FEATURE_TEMPLATE is None:
        raise ValueError("Model not loaded")
    
    # Use pre-allocated numpy array for maximum speed
    features = NUMPY_FEATURE_TEMPLATE.copy()

    # Convert to dict once
    customer_dict = customer.dict()

    # Direct numpy array assignment using cached indices
    for col in MODEL_ARTIFACTS["numeric_features"]:
        if col in FEATURE_COLUMNS_INDICES and col in customer_dict:
            idx = FEATURE_COLUMNS_INDICES[col]
            features[0, idx] = (
                customer_dict[col] if customer_dict[col] is not None else 0
            )

    # Encode categorical features with caching
    for col in MODEL_ARTIFACTS["categorical_features"]:
        encoded_col = col + "_encoded"
        if encoded_col in FEATURE_COLUMNS_INDICES and col in customer_dict:
            value = str(customer_dict[col]) if customer_dict[col] else "Unknown"

            # Use cached encoding
            cache_key = f"{col}_{value}"
            if cache_key in FEATURE_CACHE:
                encoded_value = FEATURE_CACHE[cache_key]
            else:
                # Encode and cache
                if col in MODEL_ARTIFACTS["label_encoders"]:
                    try:
                        encoded_value = MODEL_ARTIFACTS["label_encoders"][
                            col
                        ].transform([value])[0]
                        FEATURE_CACHE[cache_key] = encoded_value
                    except ValueError:
                        encoded_value = 0
                        FEATURE_CACHE[cache_key] = 0
                else:
                    encoded_value = 0

            idx = FEATURE_COLUMNS_INDICES[encoded_col]
            features[0, idx] = encoded_value

    # Add interaction features using cached indices
    if "tenure_monthly_interaction" in FEATURE_COLUMNS_INDICES:
        tenure_idx = FEATURE_COLUMNS_INDICES.get("tenure", -1)
        charges_idx = FEATURE_COLUMNS_INDICES.get("MonthlyCharges", -1)
        if tenure_idx >= 0 and charges_idx >= 0:
            interaction_idx = FEATURE_COLUMNS_INDICES["tenure_monthly_interaction"]
            features[0, interaction_idx] = (
                features[0, tenure_idx] * features[0, charges_idx]
            )

    if "charges_per_tenure" in FEATURE_COLUMNS_INDICES:
        tenure_idx = FEATURE_COLUMNS_INDICES.get("tenure", -1)
        total_charges_idx = FEATURE_COLUMNS_INDICES.get("TotalCharges", -1)
        if tenure_idx >= 0 and total_charges_idx >= 0:
            ratio_idx = FEATURE_COLUMNS_INDICES["charges_per_tenure"]
            features[0, ratio_idx] = features[0, total_charges_idx] / (
                features[0, tenure_idx] + 1
            )

    # Scale features
    if MODEL_ARTIFACTS.get("scaler") is not None:
        scaler = MODEL_ARTIFACTS["scaler"]
        features = scaler.transform(features)

    return features


def get_risk_segment(probability: float) -> str:
    """Determine risk segment based on churn probability."""
    if probability >= 0.7:
        return "Critical"
    elif probability >= 0.5:
        return "High Risk"
    elif probability >= 0.3:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_recommended_action(
    probability: float, contract: str, total_services: int = 3
) -> str:
    """Generate recommended retention action."""
    if probability >= 0.7:
        if contract == "Month-to-month":
            return "Immediate: Offer 50% discount for 1-year contract upgrade"
        else:
            return "Immediate: Personal call from account manager"
    elif probability >= 0.5:
        if total_services < 3:
            return "Proactive: Bundle discount for additional services"
        else:
            return "Proactive: Loyalty rewards program enrollment"
    elif probability >= 0.3:
        return "Monitor: Send satisfaction survey and track engagement"
    else:
        return "Maintain: Continue current service, upsell opportunities"


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Telco Customer Churn Prediction API",
        "version": "1.0.0",
        "model_loaded": MODEL_ARTIFACTS is not None,
        "documentation": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict",
        "batch_predict_endpoint": "/predict/batch",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Ultra-fast health check endpoint - no Pydantic validation."""
    uptime = (datetime.now() - APP_START_TIME).total_seconds()
    return {
        "status": "healthy" if MODEL_ARTIFACTS else "degraded",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL_ARTIFACTS is not None,
        "version": "1.0.0",
        "uptime_seconds": uptime,
    }


@app.get("/health/detailed", response_model=HealthCheckResponse, tags=["Health"])
async def health_check_detailed():
    """Detailed health check endpoint with full validation."""
    uptime = (datetime.now() - APP_START_TIME).total_seconds()
    return HealthCheckResponse(
        status="healthy" if MODEL_ARTIFACTS else "degraded",
        timestamp=datetime.now(),
        model_loaded=MODEL_ARTIFACTS is not None,
        version="1.0.0",
        uptime_seconds=uptime,
    )


@app.post("/predict", response_model=ChurnPredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn for a single customer.

    Returns churn probability, risk segment, and recommended action.
    """
    if not MODEL_ARTIFACTS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator.",
        )

    try:
        # Prepare features
        features = prepare_features(customer)

        # Make prediction - Convert numpy types to Python native types
        probability_np = MODEL_ARTIFACTS["model"].predict_proba(features)[0, 1]
        probability = float(probability_np)  # Convert numpy.float32 to Python float

        threshold = float(
            MODEL_ARTIFACTS["model_metrics"].get("optimal_threshold", 0.5)
        )
        prediction = "Yes" if probability >= threshold else "No"

        # Calculate confidence - ensure Python float
        confidence = float(abs(probability - 0.5) * 2)

        # Get risk segment and recommendation
        risk_segment = get_risk_segment(probability)
        recommended_action = get_recommended_action(probability, customer.Contract, 3)

        # Calculate value at risk - ensure Python float
        monthly_value_at_risk = float(customer.MonthlyCharges * probability)

        return ChurnPredictionResponse(
            churn_probability=probability,
            churn_prediction=prediction,
            risk_segment=risk_segment,
            confidence=confidence,
            monthly_value_at_risk=monthly_value_at_risk,
            recommended_action=recommended_action,
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"]
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers.

    Processes up to 1000 customers in a single request.
    """
    if not MODEL_ARTIFACTS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator.",
        )

    if len(request.customers) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 1000 customers per batch request",
        )

    start_time = time.time()
    predictions = []
    total_value_at_risk = 0.0
    high_risk_count = 0

    try:
        for i, customer in enumerate(request.customers):
            # Prepare features
            features = prepare_features(customer)

            # Make prediction - Convert numpy types to Python native types
            probability_np = MODEL_ARTIFACTS["model"].predict_proba(features)[0, 1]
            probability = float(probability_np)  # Convert numpy.float32 to Python float

            threshold = float(
                MODEL_ARTIFACTS["model_metrics"].get("optimal_threshold", 0.5)
            )
            prediction = "Yes" if probability >= threshold else "No"

            # Calculate metrics - ensure all are Python native types
            confidence = float(abs(probability - 0.5) * 2)
            risk_segment = get_risk_segment(probability)
            monthly_value_at_risk = float(customer.MonthlyCharges * probability)
            total_value_at_risk += monthly_value_at_risk

            if probability >= 0.5:
                high_risk_count += 1

            # Get recommendation if requested
            recommended_action = ""
            if request.include_recommendations:
                recommended_action = get_recommended_action(
                    probability, customer.Contract, 3
                )

            predictions.append(
                ChurnPredictionResponse(
                    customer_id=f"CUST_{i+1:04d}",
                    churn_probability=probability,
                    churn_prediction=prediction,
                    risk_segment=risk_segment,
                    confidence=confidence,
                    monthly_value_at_risk=monthly_value_at_risk,
                    recommended_action=recommended_action,
                )
            )

        # Calculate summary statistics - all as Python native types
        import numpy as np  # Import only when needed

        avg_churn_prob = float(np.mean([p.churn_probability for p in predictions]))
        processing_time = float(time.time() - start_time)

        return BatchPredictionResponse(
            predictions=predictions,
            summary={
                "total_customers": len(predictions),
                "high_risk_count": high_risk_count,
                "total_value_at_risk": round(total_value_at_risk, 2),
                "average_churn_probability": round(avg_churn_prob, 3),
            },
            processing_time_seconds=round(processing_time, 3),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get("/model/metrics", response_model=ModelMetricsResponse, tags=["Model"])
async def get_model_metrics():
    """
    Get model performance metrics and metadata.

    Returns model version, training metrics, and feature importance.
    """
    if not MODEL_ARTIFACTS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please contact administrator.",
        )

    try:
        # Load feature importance if available
        feature_importance = []
        if os.path.exists("models/saved/feature_importance.csv"):
            import pandas as pd

            fi_df = pd.read_csv("models/saved/feature_importance.csv")
            feature_importance = fi_df.head(10).to_dict("records")

        return ModelMetricsResponse(
            model_version="1.0.0",
            training_date=MODEL_ARTIFACTS["model_metrics"].get(
                "training_date", "Unknown"
            ),
            performance_metrics={
                "auc_roc": MODEL_ARTIFACTS["model_metrics"].get("auc", 0.0),
                "accuracy": MODEL_ARTIFACTS["model_metrics"].get("accuracy", 0.0),
                "optimal_threshold": MODEL_ARTIFACTS["model_metrics"].get(
                    "optimal_threshold", 0.5
                ),
            },
            feature_importance=feature_importance,
            data_statistics={
                "features_used": len(MODEL_ARTIFACTS["feature_columns"]),
                "numeric_features": len(MODEL_ARTIFACTS["numeric_features"]),
                "categorical_features": len(MODEL_ARTIFACTS["categorical_features"]),
            },
        )

    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model metrics: {str(e)}",
        )


@app.post("/model/reload", tags=["Model"])
async def reload_model():
    """
    Reload the model from disk.

    Useful for updating the model without restarting the service.
    """
    global MODEL_ARTIFACTS

    try:
        import joblib
        import numpy as np

        model_path = "models/saved/churn_model_artifacts.pkl"
        if os.path.exists(model_path):
            MODEL_ARTIFACTS = joblib.load(model_path)

            # Re-optimize caches
            global FEATURE_COLUMNS_INDICES, NUMPY_FEATURE_TEMPLATE
            FEATURE_COLUMNS_INDICES = {}  # Reset the dictionary
            for i, col in enumerate(MODEL_ARTIFACTS["feature_columns"]):
                FEATURE_COLUMNS_INDICES[col] = i
            NUMPY_FEATURE_TEMPLATE = np.zeros(
                (1, len(MODEL_ARTIFACTS["feature_columns"]))
            )

            logger.info("Model reloaded and re-optimized successfully")
            return {"message": "Model reloaded successfully", "status": "success"}
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        logger.error(f"Failed to reload model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(e)}",
        )


@app.get("/debug/status", tags=["Debug"])
async def debug_status():
    """Debug endpoint to check API status."""
    return {
        "model_loaded": MODEL_ARTIFACTS is not None,
        "model_type": str(type(MODEL_ARTIFACTS["model"])) if MODEL_ARTIFACTS else None,
        "feature_count": len(MODEL_ARTIFACTS["feature_columns"])
        if MODEL_ARTIFACTS
        else 0,
        "cache_size": len(FEATURE_CACHE),
        "uptime_seconds": (datetime.now() - APP_START_TIME).total_seconds(),
    }


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url),
        },
    )


# ============================================================================
# Run the application
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Environment-based configuration for production safety
    host = os.getenv("API_HOST", "127.0.0.1")  # Default localhost for dev
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))

    # Use 0.0.0.0 only in production/docker, localhost for local dev
    if os.getenv("ENV") == "production":
        host = "0.0.0.0"  # Allow external access in production

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        log_level="info",
        workers=workers if os.getenv("ENV") == "production" else 1,
    )
