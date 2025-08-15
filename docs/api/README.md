# 🔌 Telco Customer Intelligence API Documentation

## Overview

The Telco Customer Intelligence API provides RESTful endpoints for real-time customer churn prediction, batch processing, and model performance monitoring.

**Base URL**: `https://api.telco-intelligence.com/v1`  
**Local Development**: `http://localhost:8000`

## Authentication

All API requests require authentication using Bearer tokens.

```http
Authorization: Bearer YOUR_API_KEY
```

To obtain an API key:
1. Register at [developer portal](#)
2. Generate API key from dashboard
3. Include in request headers

## Rate Limiting

| Plan | Requests/Min | Requests/Day | Batch Size |
|------|-------------|--------------|------------|
| Free | 60 | 1,000 | 100 |
| Basic | 300 | 10,000 | 500 |
| Pro | 1,000 | 100,000 | 5,000 |
| Enterprise | Unlimited | Unlimited | Unlimited |

## Endpoints

### Core Endpoints

#### 🟢 GET /health

Check API health status.

**Request:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "model_version": "xgboost_v2.3",
  "last_training": "2025-01-14T08:00:00Z"
}
```

---

#### 🔮 POST /predict

Predict churn probability for a single customer.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 65.50,
    "TotalCharges": 786.00,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "InternetService": "Fiber optic",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "PaperlessBilling": "Yes",
    "Partner": "No",
    "Dependents": "No",
    "SeniorCitizen": 0,
    "gender": "Male"
  }'
```

**Response:**
```json
{
  "prediction_id": "pred_123456789",
  "timestamp": "2025-01-15T10:30:45Z",
  "customer_info": {
    "customer_id": "CUST-001",
    "tenure_months": 12,
    "monthly_charges": 65.50,
    "contract_type": "Month-to-month"
  },
  "prediction": {
    "churn_probability": 0.732,
    "churn_prediction": "Yes",
    "confidence_score": 0.85,
    "risk_segment": "High Risk",
    "percentile_rank": 92
  },
  "feature_importance": {
    "top_factors": [
      {"feature": "Contract_Month-to-month", "impact": 0.35},
      {"feature": "PaymentMethod_Electronic", "impact": 0.28},
      {"feature": "tenure", "impact": -0.22}
    ]
  },
  "recommendations": {
    "priority": "HIGH",
    "actions": [
      {
        "action": "Contract Upgrade",
        "description": "Offer 40% discount for annual contract",
        "expected_impact": 0.45,
        "implementation": "immediate"
      },
      {
        "action": "Payment Migration",
        "description": "Switch to automatic payment with $5/month discount",
        "expected_impact": 0.25,
        "implementation": "within_week"
      }
    ],
    "retention_score": 0.68
  },
  "financial_impact": {
    "monthly_revenue_at_risk": 65.50,
    "lifetime_value_at_risk": 1572.00,
    "intervention_cost": 25.00,
    "expected_roi": 3.4
  }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `prediction_id` | string | Unique identifier for this prediction |
| `churn_probability` | float | Probability of churn (0-1) |
| `churn_prediction` | string | Binary prediction (Yes/No) |
| `confidence_score` | float | Model confidence (0-1) |
| `risk_segment` | string | Risk category (Low/Medium/High/Critical) |
| `top_factors` | array | Features contributing most to prediction |
| `recommendations` | object | Personalized retention strategies |

---

#### 📦 POST /predict/batch

Process multiple customer predictions in a single request.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@customers.csv" \
  -F "include_recommendations=true" \
  -F "return_format=json"
```

**CSV Format:**
```csv
customerID,tenure,MonthlyCharges,TotalCharges,Contract,PaymentMethod,...
CUST-001,12,65.50,786.00,Month-to-month,Electronic check,...
CUST-002,24,45.25,1086.00,One year,Bank transfer,...
```

**Response:**
```json
{
  "batch_id": "batch_987654321",
  "timestamp": "2025-01-15T10:35:00Z",
  "summary": {
    "total_customers": 100,
    "processing_time_ms": 450,
    "high_risk_count": 23,
    "medium_risk_count": 31,
    "low_risk_count": 46,
    "average_churn_probability": 0.265
  },
  "predictions": [
    {
      "customer_id": "CUST-001",
      "churn_probability": 0.732,
      "risk_segment": "High Risk",
      "recommended_action": "Immediate intervention"
    },
    {
      "customer_id": "CUST-002",
      "churn_probability": 0.156,
      "risk_segment": "Low Risk",
      "recommended_action": "Standard monitoring"
    }
  ],
  "download_url": "https://api.telco-intelligence.com/results/batch_987654321.csv"
}
```

---

#### 📊 GET /model/metrics

Retrieve current model performance metrics.

**Request:**
```bash
curl -X GET "http://localhost:8000/model/metrics" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "model_info": {
    "name": "XGBoost_ChurnPredictor",
    "version": "2.3.0",
    "training_date": "2025-01-14T08:00:00Z",
    "training_samples": 4930,
    "validation_samples": 2113
  },
  "performance_metrics": {
    "auc_roc": 0.84,
    "precision": 0.41,
    "recall": 0.78,
    "f1_score": 0.54,
    "accuracy": 0.79,
    "log_loss": 0.42
  },
  "confusion_matrix": {
    "true_positive": 412,
    "false_positive": 243,
    "true_negative": 1256,
    "false_negative": 202
  },
  "feature_importance": [
    {"feature": "Contract", "importance": 0.25},
    {"feature": "tenure", "importance": 0.20},
    {"feature": "MonthlyCharges", "importance": 0.15}
  ],
  "business_metrics": {
    "customers_saved": 412,
    "revenue_retained": 487000,
    "false_alarm_rate": 0.162,
    "missed_churn_rate": 0.134
  }
}
```

---

#### 👤 GET /customer/{customer_id}/history

Retrieve prediction history for a specific customer.

**Request:**
```bash
curl -X GET "http://localhost:8000/customer/CUST-001/history" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "customer_id": "CUST-001",
  "current_status": "Active",
  "risk_trajectory": "Increasing",
  "prediction_history": [
    {
      "date": "2025-01-15",
      "churn_probability": 0.732,
      "risk_segment": "High Risk"
    },
    {
      "date": "2024-12-15",
      "churn_probability": 0.645,
      "risk_segment": "Medium Risk"
    },
    {
      "date": "2024-11-15",
      "churn_probability": 0.523,
      "risk_segment": "Medium Risk"
    }
  ],
  "interventions": [
    {
      "date": "2024-12-20",
      "type": "Retention Call",
      "outcome": "Pending",
      "notes": "Customer expressed concerns about pricing"
    }
  ]
}
```

---

### Analytics Endpoints

#### 📈 GET /analytics/segments

Get customer segmentation analysis.

**Request:**
```bash
curl -X GET "http://localhost:8000/analytics/segments" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Response:**
```json
{
  "segments": [
    {
      "name": "Critical Risk",
      "count": 250,
      "percentage": 3.5,
      "avg_churn_probability": 0.85,
      "avg_monthly_charges": 89.50,
      "total_revenue_at_risk": 22375.00
    },
    {
      "name": "High Risk",
      "count": 1619,
      "percentage": 23.0,
      "avg_churn_probability": 0.65,
      "avg_monthly_charges": 78.90,
      "total_revenue_at_risk": 127739.10
    }
  ]
}
```

---

#### 💰 POST /analytics/roi

Calculate ROI for retention campaigns.

**Request:**
```bash
curl -X POST "http://localhost:8000/analytics/roi" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "target_segment": "High Risk",
    "intervention_type": "contract_upgrade",
    "intervention_cost": 25,
    "success_rate": 0.3,
    "discount_percentage": 20
  }'
```

**Response:**
```json
{
  "campaign_analysis": {
    "target_customers": 1619,
    "expected_conversions": 486,
    "prevented_churns": 316,
    "campaign_cost": 40475,
    "revenue_saved": 298440,
    "net_benefit": 257965,
    "roi_percentage": 637,
    "payback_period_days": 42
  }
}
```

---

### Administrative Endpoints

#### ⚙️ POST /model/retrain

Trigger model retraining with new data.

**Request:**
```bash
curl -X POST "http://localhost:8000/model/retrain" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "training_data": "s3://bucket/new_data.csv",
    "model_type": "xgboost",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 6,
      "learning_rate": 0.05
    }
  }'
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required field: tenure",
    "details": {
      "field": "tenure",
      "requirement": "integer between 0 and 72"
    },
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request or missing fields |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Webhooks

Configure webhooks to receive real-time notifications.

### Webhook Events

- `prediction.completed` - Single prediction processed
- `batch.completed` - Batch processing finished
- `high_risk.detected` - Critical risk customer identified
- `model.retrained` - Model updated

### Webhook Payload

```json
{
  "event": "high_risk.detected",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "customer_id": "CUST-001",
    "churn_probability": 0.92,
    "recommended_action": "immediate_intervention"
  }
}
```

## SDK & Libraries

### Python SDK

```python
from telco_intelligence import TelcoAPI

# Initialize client
client = TelcoAPI(api_key="YOUR_API_KEY")

# Single prediction
result = client.predict(
    tenure=12,
    monthly_charges=65.50,
    contract="Month-to-month"
)

print(f"Churn probability: {result.churn_probability}")
print(f"Recommended action: {result.recommendations[0]}")

# Batch prediction
results = client.predict_batch("customers.csv")
for prediction in results:
    print(f"{prediction.customer_id}: {prediction.risk_segment}")
```

### JavaScript SDK

```javascript
const TelcoAPI = require('telco-intelligence');

const client = new TelcoAPI({ apiKey: 'YOUR_API_KEY' });

// Single prediction
const result = await client.predict({
  tenure: 12,
  monthlyCharges: 65.50,
  contract: 'Month-to-month'
});

console.log(`Churn probability: ${result.churnProbability}`);
```

## Testing

### Test Environment

Base URL: `https://sandbox.telco-intelligence.com/v1`  
Test API Key: `test_key_123456789`

### Sample Test Data

```json
{
  "test_high_risk": {
    "tenure": 2,
    "MonthlyCharges": 95.00,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
  },
  "test_low_risk": {
    "tenure": 60,
    "MonthlyCharges": 45.00,
    "Contract": "Two year",
    "PaymentMethod": "Bank transfer (automatic)"
  }
}
```

## Support

- 📧 Email: api-support@telco-intelligence.com
- 📚 Documentation: https://docs.telco-intelligence.com
- 💬 Slack: telco-intelligence.slack.com
- 🐛 Issues: github.com/telco-intelligence/api/issues