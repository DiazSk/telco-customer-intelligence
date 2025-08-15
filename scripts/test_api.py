"""
Test script for the Telco Churn Prediction API.

Location: scripts/test_api.py
Run with: python scripts/test_api.py
"""

import json
import time
from typing import Dict

import requests


class APITester:
    """Test client for the Churn Prediction API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API tester."""
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}

    def test_health_check(self):
        """Test health check endpoint."""
        print("\n" + "=" * 60)
        print("Testing Health Check Endpoint")
        print("=" * 60)

        response = requests.get(f"{self.base_url}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200

    def test_single_prediction(self):
        """Test single customer prediction."""
        print("\n" + "=" * 60)
        print("Testing Single Prediction Endpoint")
        print("=" * 60)

        # Test customer data
        customer_data = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,  # New customer - high risk
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",  # High risk contract
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",  # High risk payment
            "MonthlyCharges": 85.70,
            "TotalCharges": 85.70,
        }

        print("Customer Profile:")
        print(f"  - Tenure: {customer_data['tenure']} months (New Customer)")
        print(f"  - Contract: {customer_data['Contract']}")
        print(f"  - Payment: {customer_data['PaymentMethod']}")
        print(f"  - Monthly Charges: ${customer_data['MonthlyCharges']}")

        response = requests.post(
            f"{self.base_url}/predict", json=customer_data, headers=self.headers
        )

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Results:")
            print(f"  - Churn Probability: {result['churn_probability']:.2%}")
            print(f"  - Prediction: {result['churn_prediction']}")
            print(f"  - Risk Segment: {result['risk_segment']}")
            print(f"  - Monthly Value at Risk: ${result['monthly_value_at_risk']:.2f}")
            print(f"  - Recommended Action: {result['recommended_action']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    def test_batch_prediction(self):
        """Test batch prediction endpoint."""
        print("\n" + "=" * 60)
        print("Testing Batch Prediction Endpoint")
        print("=" * 60)

        # Create different customer profiles
        customers = [
            {  # High risk - new customer
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.0,
                "TotalCharges": 140.0,
            },
            {  # Low risk - loyal customer
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 60,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 95.0,
                "TotalCharges": 5700.0,
            },
            {  # Medium risk
                "gender": "Female",
                "SeniorCitizen": 1,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 50.0,
                "TotalCharges": 1200.0,
            },
        ]

        batch_request = {"customers": customers, "include_recommendations": True}

        print(f"Sending {len(customers)} customers for batch prediction...")

        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/predict/batch", json=batch_request, headers=self.headers
        )
        elapsed_time = time.time() - start_time

        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed_time:.3f} seconds")

        if response.status_code == 200:
            result = response.json()
            print("\nBatch Prediction Summary:")
            print(f"  - Total Customers: {result['summary']['total_customers']}")
            print(f"  - High Risk Count: {result['summary']['high_risk_count']}")
            print(f"  - Total Value at Risk: ${result['summary']['total_value_at_risk']:.2f}")
            avg_prob = result["summary"]["average_churn_probability"]
            print(f"  - Average Churn Probability: {avg_prob:.2%}")
            print(f"  - Processing Time: {result['processing_time_seconds']} seconds")

            print("\nIndividual Predictions:")
            for i, pred in enumerate(result["predictions"], 1):
                risk = pred["risk_segment"]
                prob = pred["churn_probability"]
                print(f"  Customer {i}: {risk} ({prob:.2%})")

            return True
        else:
            print(f"Error: {response.text}")
            return False

    def test_model_metrics(self):
        """Test model metrics endpoint."""
        print("\n" + "=" * 60)
        print("Testing Model Metrics Endpoint")
        print("=" * 60)

        response = requests.get(f"{self.base_url}/model/metrics")
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\nModel Information:")
            print(f"  - Version: {result['model_version']}")
            print(f"  - Training Date: {result['training_date']}")

            print("\nPerformance Metrics:")
            for metric, value in result["performance_metrics"].items():
                print(f"  - {metric}: {value:.3f}")

            print("\nTop Features:")
            for i, feature in enumerate(result["feature_importance"][:5], 1):
                print(f"  {i}. {feature['feature']}: {feature['importance']:.3f}")

            return True
        else:
            print(f"Error: {response.text}")
            return False

    def test_invalid_request(self):
        """Test API error handling."""
        print("\n" + "=" * 60)
        print("Testing Error Handling")
        print("=" * 60)

        # Invalid customer data (missing required fields)
        invalid_data = {
            "gender": "Male",
            "tenure": 12,
            # Missing many required fields
        }

        print("Sending invalid request (missing required fields)...")
        response = requests.post(
            f"{self.base_url}/predict", json=invalid_data, headers=self.headers
        )

        print(f"Status Code: {response.status_code}")
        print(f"Error Response: {json.dumps(response.json(), indent=2)}")

        # Expecting 422 Unprocessable Entity
        return response.status_code == 422

    def test_performance(self):
        """Test API performance with multiple requests."""
        print("\n" + "=" * 60)
        print("Testing API Performance")
        print("=" * 60)

        # Sample customer for testing
        customer_data = {
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
            "TotalCharges": 1320.0,
        }

        num_requests = 10
        response_times = []

        print(f"Sending {num_requests} sequential requests...")

        for i in range(num_requests):
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/predict", json=customer_data, headers=self.headers
            )
            elapsed = time.time() - start_time
            response_times.append(elapsed)

            if response.status_code != 200:
                print(f"Request {i+1} failed!")
                return False

        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)

        print("\nPerformance Results:")
        print(f"  - Average Response Time: {avg_time*1000:.2f} ms")
        print(f"  - Min Response Time: {min_time*1000:.2f} ms")
        print(f"  - Max Response Time: {max_time*1000:.2f} ms")
        print(f"  - Requests per Second: {1/avg_time:.1f}")

        # Check if average response time is under 100ms
        if avg_time < 0.1:
            print("  ✅ Performance target met (<100ms)")
        else:
            print("  ⚠️ Performance below target (>100ms)")

        return True

    def run_all_tests(self):
        """Run all API tests."""
        print("\n" + "🚀 Starting API Tests " + "=" * 35)

        tests = [
            ("Health Check", self.test_health_check),
            ("Single Prediction", self.test_single_prediction),
            ("Batch Prediction", self.test_batch_prediction),
            ("Model Metrics", self.test_model_metrics),
            ("Error Handling", self.test_invalid_request),
            ("Performance", self.test_performance),
        ]

        results = []

        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
                time.sleep(0.5)  # Small delay between tests
            except Exception as e:
                print(f"\n❌ Test '{test_name}' failed with error: {str(e)}")
                results.append((test_name, False))

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        for test_name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{test_name}: {status}")

        passed = sum(1 for _, success in results if success)
        total = len(results)

        print(f"\nTotal: {passed}/{total} tests passed")

        if passed == total:
            print("\n🎉 All tests passed successfully!")
        else:
            print(f"\n⚠️ {total - passed} test(s) failed. Please review the output above.")

        return passed == total


def main():
    """Main test runner."""
    print("=" * 60)
    print("TELCO CHURN PREDICTION API - TEST SUITE")
    print("=" * 60)

    # Check if API is running
    print("\nChecking if API is running at http://localhost:8000...")
    try:
        response = requests.get("http://localhost:8000")
        print("✅ API is running!")
    except requests.exceptions.ConnectionError:
        print("❌ API is not running!")
        print("\nPlease start the API first with:")
        print("  uvicorn src.api.main:app --reload --port 8000")
        return

    # Create tester and run tests
    tester = APITester()
    success = tester.run_all_tests()

    if success:
        print("\n✅ API is ready for production!")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before deployment.")


if __name__ == "__main__":
    main()