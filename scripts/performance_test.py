"""
Performance debugging script.

Save as: scripts/performance_test.py
Run: python scripts/performance_test.py
"""

import time
import requests
import json
import statistics

def test_endpoint_performance(endpoint, data, num_requests=5):
    """Test a specific endpoint's performance."""
    times = []
    
    for i in range(num_requests):
        start = time.time()
        response = requests.post(endpoint, json=data)
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # Convert to ms
        
        if i == 0:
            print(f"First request: {elapsed*1000:.2f}ms")
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                return None
    
    return {
        "avg": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "median": statistics.median(times)
    }

def main():
    """Run performance tests."""
    print("=" * 60)
    print("API PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Test data
    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 55.0,
        "TotalCharges": 1320.0
    }
    
    # Test health endpoint (baseline)
    print("\n1. Testing Health Endpoint (baseline):")
    start = time.time()
    response = requests.get("http://127.0.0.1:8000/health")
    health_time = (time.time() - start) * 1000
    print(f"   Health check: {health_time:.2f}ms")
    
    # Test single prediction
    print("\n2. Testing Single Prediction:")
    stats = test_endpoint_performance(
        "http://127.0.0.1:8000/predict",
        customer,
        num_requests=5
    )
    
    if stats:
        print(f"   Average: {stats['avg']:.2f}ms")
        print(f"   Min: {stats['min']:.2f}ms")
        print(f"   Max: {stats['max']:.2f}ms")
        print(f"   Median: {stats['median']:.2f}ms")
    
    # Performance verdict
    print("\n" + "=" * 60)
    print("PERFORMANCE VERDICT:")
    if stats and stats['avg'] < 100:
        print("✅ Performance GOOD (<100ms)")
    elif stats and stats['avg'] < 500:
        print("⚠️ Performance ACCEPTABLE (100-500ms)")
    else:
        print("❌ Performance POOR (>500ms)")
        print("\nLikely issues:")
        print("  1. Model loading on each request")
        print("  2. Inefficient feature preparation")
        print("  3. Large model size")
        print("  4. Missing caching")
    
    print("=" * 60)

if __name__ == "__main__":
    main()