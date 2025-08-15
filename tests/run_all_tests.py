#!/usr/bin/env python
"""
Test Runner for Telco Customer Intelligence Platform
Runs all tests in the proper order with comprehensive reporting
"""

import subprocess
import sys
import os
from pathlib import Path

class TestRunner:
    def __init__(self):
        self.test_root = Path(__file__).parent
        self.project_root = self.test_root.parent
        self.results = {}
        
    def print_banner(self):
        """Print test runner banner"""
        print("""
        ╔════════════════════════════════════════════════════════════╗
        ║                                                            ║
        ║     🧪 TELCO INTELLIGENCE PLATFORM - TEST SUITE 🧪       ║
        ║                                                            ║
        ║     Comprehensive testing across all components           ║
        ║                                                            ║
        ╚════════════════════════════════════════════════════════════╝
        """)
        
    def run_unit_tests(self):
        """Run unit tests"""
        print("\n🔬 Running Unit Tests...")
        print("=" * 60)
        
        unit_tests = [
            "tests/unit/test_pipeline.py",
            "tests/unit/test_dashboard.py"
        ]
        
        for test in unit_tests:
            if os.path.exists(test):
                print(f"\n📋 Running: {test}")
                try:
                    result = subprocess.run([sys.executable, test], 
                                          capture_output=True, text=True, cwd=str(self.project_root))
                    if result.returncode == 0:
                        print("✅ PASSED")
                        self.results[test] = "PASSED"
                    else:
                        print("❌ FAILED")
                        print(result.stdout)
                        print(result.stderr)
                        self.results[test] = "FAILED"
                except Exception as e:
                    print(f"❌ ERROR: {e}")
                    self.results[test] = "ERROR"
            else:
                print(f"⚠️ Test file not found: {test}")
                
    def run_integration_tests(self):
        """Run integration tests"""
        print("\n🔗 Running Integration Tests...")
        print("=" * 60)
        
        integration_tests = [
            "tests/integration/test_api.py",
            "tests/integration/test_dashboard_integration.py"
        ]
        
        for test in integration_tests:
            if os.path.exists(test):
                print(f"\n📋 Running: {test}")
                try:
                    result = subprocess.run([sys.executable, test], 
                                          capture_output=True, text=True, cwd=str(self.project_root))
                    if result.returncode == 0:
                        print("✅ PASSED")
                        self.results[test] = "PASSED"
                    else:
                        print("❌ FAILED")
                        print(result.stdout)
                        print(result.stderr)
                        self.results[test] = "FAILED"
                except Exception as e:
                    print(f"❌ ERROR: {e}")
                    self.results[test] = "ERROR"
            else:
                print(f"⚠️ Test file not found: {test}")
                
    def run_performance_tests(self):
        """Run performance tests"""
        print("\n⚡ Running Performance Tests...")
        print("=" * 60)
        
        performance_tests = [
            "tests/performance/performance_test.py"
        ]
        
        for test in performance_tests:
            if os.path.exists(test):
                print(f"\n📋 Running: {test}")
                try:
                    result = subprocess.run([sys.executable, test], 
                                          capture_output=True, text=True, cwd=str(self.project_root))
                    if result.returncode == 0:
                        print("✅ PASSED")
                        self.results[test] = "PASSED"
                    else:
                        print("❌ FAILED")
                        print(result.stdout)
                        print(result.stderr)
                        self.results[test] = "FAILED"
                except Exception as e:
                    print(f"❌ ERROR: {e}")
                    self.results[test] = "ERROR"
            else:
                print(f"⚠️ Test file not found: {test}")
                
    def run_validation_tests(self):
        """Run validation tests"""
        print("\n✅ Running Validation Tests...")
        print("=" * 60)
        
        validation_tests = [
            "tests/validation/validate_pro_tips.py"
        ]
        
        for test in validation_tests:
            if os.path.exists(test):
                print(f"\n📋 Running: {test}")
                try:
                    result = subprocess.run([sys.executable, test], 
                                          capture_output=True, text=True, cwd=str(self.project_root))
                    if result.returncode == 0:
                        print("✅ PASSED")
                        self.results[test] = "PASSED"
                    else:
                        print("❌ FAILED")
                        print(result.stdout)
                        print(result.stderr)
                        self.results[test] = "FAILED"
                except Exception as e:
                    print(f"❌ ERROR: {e}")
                    self.results[test] = "ERROR"
            else:
                print(f"⚠️ Test file not found: {test}")
                
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.results.values() if result == "PASSED")
        failed = sum(1 for result in self.results.values() if result == "FAILED")
        errors = sum(1 for result in self.results.values() if result == "ERROR")
        total = len(self.results)
        
        print(f"\n📈 Results:")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   🚨 Errors: {errors}")
        print(f"   📊 Total: {total}")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"   🎯 Success Rate: {success_rate:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        for test, result in self.results.items():
            status = {"PASSED": "✅", "FAILED": "❌", "ERROR": "🚨"}[result]
            print(f"   {status} {test}")
            
        if passed == total:
            print(f"\n🎉 ALL TESTS PASSED! Your platform is ready for deployment!")
        elif failed > 0:
            print(f"\n⚠️ Some tests failed. Please review and fix issues before deployment.")
        else:
            print(f"\n💡 Tests completed with some issues. Review the results above.")
            
    def run_all(self):
        """Run all test suites"""
        self.print_banner()
        
        # Change to project root for test execution
        os.chdir(str(self.project_root))
        
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_performance_tests()
        self.run_validation_tests()
        
        self.print_summary()

def main():
    """Main test runner function"""
    runner = TestRunner()
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "unit":
            runner.run_unit_tests()
        elif test_type == "integration":
            runner.run_integration_tests()
        elif test_type == "performance":
            runner.run_performance_tests()
        elif test_type == "validation":
            runner.run_validation_tests()
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python run_all_tests.py [unit|integration|performance|validation]")
    else:
        runner.run_all()

if __name__ == "__main__":
    main()
