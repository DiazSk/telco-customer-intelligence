#!/usr/bin/env python
"""
Complete Dashboard Integration Test
Run this to verify your Day 4 dashboard implementation
"""

import os
import sys
import time
import requests
import pandas as pd
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class DashboardTester:
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.dashboard_url = "http://localhost:8501"
        self.test_results = []
        
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"🧪 {text}")
        print("="*60)
        
    def test_file_structure(self):
        """Test if all required files exist"""
        self.print_header("Testing File Structure")
        
        required_files = [
            "src/dashboard/app.py",
            "src/dashboard/dashboard_utils.py",
            "src/dashboard/__init__.py",
            "data/processed/processed_telco_data.csv",
            "data/processed/customer_segments.csv"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path} exists")
                self.test_results.append(("File Structure", file_path, True))
            else:
                print(f"❌ {file_path} missing")
                self.test_results.append(("File Structure", file_path, False))
                
    def test_data_loading(self):
        """Test if data files can be loaded"""
        self.print_header("Testing Data Loading")
        
        try:
            # Load main data
            df = pd.read_csv("data/processed/processed_telco_data.csv")
            print(f"✅ Loaded main data: {len(df)} records")
            self.test_results.append(("Data Loading", "Main dataset", True))
            
            # Check for required columns
            required_cols = ['customerID', 'Churn', 'MonthlyCharges', 'TotalCharges', 'Contract']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️  Missing columns: {missing_cols}")
                self.test_results.append(("Data Columns", "Required columns", False))
            else:
                print(f"✅ All required columns present")
                self.test_results.append(("Data Columns", "Required columns", True))
                
            # Load segments data
            segments = pd.read_csv("data/processed/customer_segments.csv")
            print(f"✅ Loaded segments: {len(segments)} records")
            self.test_results.append(("Data Loading", "Segments dataset", True))
            
            # Check segment columns
            if 'risk_segment' in segments.columns:
                print(f"✅ Risk segments found: {segments['risk_segment'].unique()}")
                self.test_results.append(("Data Columns", "Risk segments", True))
            else:
                print(f"⚠️  Risk segment column missing")
                self.test_results.append(("Data Columns", "Risk segments", False))
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.test_results.append(("Data Loading", "Error", False))
            
    def test_api_connection(self):
        """Test API connectivity"""
        self.print_header("Testing API Connection")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API is running at {self.api_url}")
                self.test_results.append(("API", "Health check", True))
            else:
                print(f"⚠️  API returned status {response.status_code}")
                self.test_results.append(("API", "Health check", False))
                
            # Test prediction endpoint
            test_customer = {
                # Demographics (required)
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes", 
                "Dependents": "No",
                
                # Account information (required)
                "tenure": 12,
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                
                # Services (required)
                "PhoneService": "Yes",
                "InternetService": "Fiber optic",
                
                # Optional services (with defaults)
                "MultipleLines": "No",
                "OnlineSecurity": "No",
                "OnlineBackup": "No", 
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                
                # Charges (required)
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0
            }
            
            response = requests.post(
                f"{self.api_url}/predict",
                json=test_customer,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Prediction endpoint works")
                print(f"   - Churn probability: {result.get('churn_probability', 'N/A')}")
                print(f"   - Risk segment: {result.get('risk_segment', 'N/A')}")
                self.test_results.append(("API", "Prediction endpoint", True))
            else:
                print(f"⚠️  Prediction endpoint returned {response.status_code}")
                self.test_results.append(("API", "Prediction endpoint", False))
                
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to API at {self.api_url}")
            print(f"   Please start the API with: python src/api/main.py")
            self.test_results.append(("API", "Connection", False))
        except Exception as e:
            print(f"❌ API test error: {e}")
            self.test_results.append(("API", "Error", False))
            
    def test_dashboard_utilities(self):
        """Test dashboard utility functions"""
        self.print_header("Testing Dashboard Utilities")
        
        try:
            from src.dashboard.dashboard_utils import (
                format_currency,
                format_percentage,
                calculate_business_metrics,
                calculate_intervention_roi
            )
            
            # Test formatting functions
            assert format_currency(1500000) == "$1.5M"
            assert format_percentage(0.256) == "25.6%"
            print("✅ Formatting utilities work")
            self.test_results.append(("Utils", "Formatting", True))
            
            # Test business metrics
            df = pd.read_csv("data/processed/processed_telco_data.csv")
            metrics = calculate_business_metrics(df)
            
            required_metrics = ['total_customers', 'churn_rate', 'monthly_revenue']
            missing_metrics = [m for m in required_metrics if m not in metrics]
            
            if missing_metrics:
                print(f"⚠️  Missing metrics: {missing_metrics}")
                self.test_results.append(("Utils", "Business metrics", False))
            else:
                print(f"✅ Business metrics calculated successfully")
                print(f"   - Total customers: {metrics['total_customers']:,}")
                print(f"   - Churn rate: {metrics['churn_rate']*100:.1f}%")
                print(f"   - Monthly revenue: ${metrics['monthly_revenue']:,.0f}")
                self.test_results.append(("Utils", "Business metrics", True))
                
            # Test ROI calculation
            roi = calculate_intervention_roi(
                num_customers=100,
                avg_churn_prob=0.3,
                intervention_cost=25,
                success_rate=0.4,
                customer_lifetime_value=2000
            )
            
            if roi['roi_percentage'] > 0:
                print(f"✅ ROI calculation works: {roi['roi_percentage']:.1f}%")
                self.test_results.append(("Utils", "ROI calculation", True))
            else:
                print(f"⚠️  ROI calculation returned negative: {roi['roi_percentage']:.1f}%")
                self.test_results.append(("Utils", "ROI calculation", True))
                
        except ImportError as e:
            print(f"❌ Cannot import dashboard utilities: {e}")
            self.test_results.append(("Utils", "Import", False))
        except Exception as e:
            print(f"❌ Utility test error: {e}")
            self.test_results.append(("Utils", "Error", False))
            
    def test_streamlit_availability(self):
        """Test if Streamlit is properly installed"""
        self.print_header("Testing Streamlit Installation")
        
        try:
            import streamlit as st
            print(f"✅ Streamlit version: {st.__version__}")
            self.test_results.append(("Streamlit", "Installation", True))
            
            import plotly
            print(f"✅ Plotly version: {plotly.__version__}")
            self.test_results.append(("Plotly", "Installation", True))
            
        except ImportError as e:
            print(f"❌ Missing package: {e}")
            print(f"   Install with: pip install streamlit plotly")
            self.test_results.append(("Dependencies", "Missing", False))
            
    def test_dashboard_config(self):
        """Test dashboard configuration"""
        self.print_header("Testing Dashboard Configuration")
        
        config_path = "src/dashboard/config.py"
        if os.path.exists(config_path):
            print(f"✅ Config file exists")
            self.test_results.append(("Config", "File exists", True))
            
            try:
                from src.dashboard.config import (
                    API_BASE_URL,
                    PAGE_TITLE,
                    RISK_COLORS
                )
                print(f"✅ Config loaded successfully")
                print(f"   - API URL: {API_BASE_URL}")
                print(f"   - Page Title: {PAGE_TITLE}")
                self.test_results.append(("Config", "Import", True))
            except ImportError as e:
                print(f"⚠️  Config import error: {e}")
                self.test_results.append(("Config", "Import", False))
        else:
            print(f"ℹ️  No config file found (optional)")
            self.test_results.append(("Config", "File exists", None))
            
    def generate_test_report(self):
        """Generate final test report"""
        self.print_header("Test Summary Report")
        
        total_tests = len(self.test_results)
        passed = sum(1 for _, _, result in self.test_results if result is True)
        failed = sum(1 for _, _, result in self.test_results if result is False)
        skipped = sum(1 for _, _, result in self.test_results if result is None)
        
        print(f"\n📊 Test Results:")
        print(f"   Total: {total_tests}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   ℹ️  Skipped: {skipped}")
        
        if failed > 0:
            print(f"\n❌ Failed Tests:")
            for category, test, result in self.test_results:
                if result is False:
                    print(f"   - {category}: {test}")
                    
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("\n✨ Great job! Your dashboard is ready to run!")
            print("   Launch with: streamlit run src/dashboard/app.py")
        elif success_rate >= 60:
            print("\n⚠️  Dashboard is mostly ready but needs some fixes.")
            print("   Review the failed tests above.")
        else:
            print("\n❌ Dashboard needs significant setup work.")
            print("   Please address the failed tests above.")
            
        # Save report
        report_path = "test_results_dashboard.json"
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_tests': total_tests,
                'passed': passed,
                'failed': failed,
                'skipped': skipped,
                'success_rate': success_rate,
                'details': self.test_results
            }, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_path}")
        
def main():
    """Run all tests"""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║     Day 4: Dashboard Integration Test Suite          ║
    ║     Testing your Telco Customer Intelligence         ║
    ║     Dashboard Implementation                         ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    tester = DashboardTester()
    
    # Run all tests
    tester.test_file_structure()
    tester.test_data_loading()
    tester.test_streamlit_availability()
    tester.test_dashboard_utilities()
    tester.test_api_connection()
    tester.test_dashboard_config()
    
    # Generate report
    tester.generate_test_report()
    
    print("\n" + "="*60)
    print("🏁 Testing Complete!")
    print("="*60)
    
    print("""
    Next Steps:
    1. Fix any failed tests (❌) shown above
    2. Start your API: python src/api/main.py
    3. Launch dashboard: streamlit run src/dashboard/app.py
    4. Open browser to: http://localhost:8501
    
    Need help? Check the implementation guide!
    """)

if __name__ == "__main__":
    main()