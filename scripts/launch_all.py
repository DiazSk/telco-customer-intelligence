#!/usr/bin/env python
"""
Launch All Services for Telco Customer Intelligence Platform
This script starts the API and Dashboard together
"""

import subprocess
import sys
import os
import time
import threading
import webbrowser
from pathlib import Path

class ServiceLauncher:
    def __init__(self):
        self.processes = []
        self.api_process = None
        self.dashboard_process = None
        
    def print_banner(self):
        """Print welcome banner"""
        print("""
        ╔════════════════════════════════════════════════════════════╗
        ║                                                            ║
        ║     🎯 TELCO CUSTOMER INTELLIGENCE PLATFORM 🎯            ║
        ║                                                            ║
        ║     Starting all services...                              ║
        ║                                                            ║
        ╚════════════════════════════════════════════════════════════╝
        """)
        
    def check_dependencies(self):
        """Check if required packages are installed"""
        print("🔍 Checking dependencies...")
        
        required_packages = ['fastapi', 'streamlit', 'pandas', 'plotly']
        missing = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package} installed")
            except ImportError:
                print(f"   ❌ {package} missing")
                missing.append(package)
                
        if missing:
            print(f"\n❌ Missing packages: {', '.join(missing)}")
            print(f"   Install with: pip install {' '.join(missing)}")
            return False
            
        return True
        
    def check_data_files(self):
        """Check if required data files exist"""
        print("\n📁 Checking data files...")
        
        required_files = [
            "data/processed/processed_telco_data.csv",
            "data/processed/customer_segments.csv"
        ]
        
        missing = []
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} missing")
                missing.append(file_path)
                
        if missing:
            print("\n❌ Missing data files. Please run the data pipeline first:")
            print("   python scripts/run_pipeline.py")
            return False
            
        return True
        
    def start_api(self):
        """Start the FastAPI server"""
        print("\n🚀 Starting API server...")
        
        try:
            # Change to API directory
            api_path = Path("src/api/main.py")
            
            if not api_path.exists():
                print("   ❌ API not found at src/api/main.py")
                return None
                
            # Start API process
            self.api_process = subprocess.Popen(
                [sys.executable, str(api_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for API to start
            print("   ⏳ Waiting for API to start...")
            time.sleep(3)
            
            # Check if API is running
            import requests
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("   ✅ API is running at http://localhost:8000")
                    print("   📖 API docs at http://localhost:8000/docs")
                    return self.api_process
            except:
                pass
                
            print("   ⚠️  API may be starting slowly...")
            return self.api_process
            
        except Exception as e:
            print(f"   ❌ Failed to start API: {e}")
            return None
            
    def start_dashboard(self):
        """Start the Streamlit dashboard"""
        print("\n🎨 Starting Dashboard...")
        
        try:
            dashboard_path = Path("src/dashboard/app.py")
            
            if not dashboard_path.exists():
                print("   ❌ Dashboard not found at src/dashboard/app.py")
                return None
                
            # Start dashboard process
            self.dashboard_process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
                 "--server.port=8501", "--server.headless=true"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print("   ⏳ Waiting for dashboard to start...")
            time.sleep(5)
            
            print("   ✅ Dashboard is running at http://localhost:8501")
            
            return self.dashboard_process
            
        except Exception as e:
            print(f"   ❌ Failed to start dashboard: {e}")
            return None
            
    def open_browser(self):
        """Open dashboard in browser"""
        time.sleep(2)
        print("\n🌐 Opening dashboard in browser...")
        webbrowser.open("http://localhost:8501")
        
    def monitor_services(self):
        """Monitor running services"""
        print("\n" + "="*60)
        print("✅ ALL SERVICES RUNNING!")
        print("="*60)
        print("""
        📍 Access Points:
        • Dashboard: http://localhost:8501
        • API: http://localhost:8000
        • API Docs: http://localhost:8000/docs
        
        📊 Available Features:
        • Executive Dashboard
        • Real-time Predictions
        • Customer Analytics
        • ROI Calculator
        • What-If Scenarios
        • Model Performance
        
        🛑 To stop all services: Press Ctrl+C
        """)
        
        try:
            # Keep running until interrupted
            while True:
                # Check if processes are still running
                if self.api_process and self.api_process.poll() is not None:
                    print("\n⚠️  API has stopped")
                    break
                if self.dashboard_process and self.dashboard_process.poll() is not None:
                    print("\n⚠️  Dashboard has stopped")
                    break
                    
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping all services...")
            self.cleanup()
            
    def cleanup(self):
        """Clean up processes"""
        if self.api_process:
            self.api_process.terminate()
            print("   ✅ API stopped")
            
        if self.dashboard_process:
            self.dashboard_process.terminate()
            print("   ✅ Dashboard stopped")
            
        print("\n👋 All services stopped. Goodbye!")
        
    def run(self):
        """Main execution"""
        self.print_banner()
        
        # Check prerequisites
        if not self.check_dependencies():
            print("\n❌ Please install missing dependencies first")
            return
            
        if not self.check_data_files():
            print("\n❌ Please prepare data files first")
            return
            
        # Start services
        api = self.start_api()
        if not api:
            print("\n❌ Failed to start API")
            return
            
        dashboard = self.start_dashboard()
        if not dashboard:
            print("\n❌ Failed to start dashboard")
            self.cleanup()
            return
            
        # Open browser
        browser_thread = threading.Thread(target=self.open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Monitor services
        self.monitor_services()

def main():
    """Entry point"""
    launcher = ServiceLauncher()
    
    try:
        launcher.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        launcher.cleanup()
    finally:
        launcher.cleanup()

if __name__ == "__main__":
    # Quick mode selection
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "api":
            print("🚀 Starting API only...")
            subprocess.run([sys.executable, "src/api/main.py"])
            
        elif mode == "dashboard":
            print("🎨 Starting Dashboard only...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"])
            
        elif mode == "test":
            print("🧪 Running tests...")
            subprocess.run([sys.executable, "tests/integration/test_dashboard_integration.py"])
            
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python launch_all.py [api|dashboard|test]")
    else:
        # Default: launch all services
        main()