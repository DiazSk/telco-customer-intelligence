#!/usr/bin/env python
"""
Launch Dashboard with Configuration
"""

import subprocess
import sys
import os
import time

def check_api_running():
    """Check if API is running"""
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Telco Customer Intelligence Dashboard...")
    
    # Check API
    if not check_api_running():
        print("⚠️  Warning: API is not running. Some features may not work.")
        print("   Start the API with: python src/api/main.py")
    else:
        print("✅ API is running")
    
    # Launch dashboard
    dashboard_path = os.path.join("src", "dashboard", "app.py")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            dashboard_path,
            "--server.port=8501",
            "--server.headless=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    launch_dashboard()