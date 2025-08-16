#!/usr/bin/env python3
"""
Docker Container Test Script
Tests Docker containers for the Telco Customer Intelligence Platform
"""

import sys
import time
import requests
import subprocess
import logging
from typing import Dict, List, Tuple
import yaml
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DockerTester:
    """Test Docker containers for the application"""
    
    def __init__(self):
        self.compose_file = "docker-compose.yml"
        self.services = {
            "postgres": {"port": 5432, "health_check": self.check_postgres},
            "redis": {"port": 6379, "health_check": self.check_redis},
            "api": {"port": 8000, "health_check": self.check_api},
            "dashboard": {"port": 8501, "health_check": self.check_dashboard},
            "mlflow": {"port": 5000, "health_check": self.check_mlflow}
        }
        
    def run_command(self, command: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
        """Run a shell command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(command)}")
            return 1, "", "Command timed out"
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return 1, "", str(e)
    
    def check_docker_compose_syntax(self) -> bool:
        """Validate docker-compose.yml syntax"""
        logger.info("Checking docker-compose.yml syntax...")
        
        try:
            with open(self.compose_file, 'r') as f:
                yaml.safe_load(f)
            logger.info("✅ docker-compose.yml syntax is valid")
            return True
        except yaml.YAMLError as e:
            logger.error(f"❌ docker-compose.yml syntax error: {e}")
            return False
        except FileNotFoundError:
            logger.error(f"❌ {self.compose_file} not found")
            return False
    
    def build_images(self) -> bool:
        """Build Docker images"""
        logger.info("Building Docker images...")
        
        # Build main application image
        exit_code, stdout, stderr = self.run_command([
            "docker", "compose", "build", "--no-cache"
        ])
        
        if exit_code != 0:
            logger.error(f"❌ Failed to build images: {stderr}")
            return False
        
        logger.info("✅ Docker images built successfully")
        return True
    
    def start_services(self) -> bool:
        """Start Docker services"""
        logger.info("Starting Docker services...")
        
        exit_code, stdout, stderr = self.run_command([
            "docker", "compose", "up", "-d"
        ])
        
        if exit_code != 0:
            logger.error(f"❌ Failed to start services: {stderr}")
            return False
        
        logger.info("✅ Services started successfully")
        return True
    
    def wait_for_services(self, timeout: int = 120) -> bool:
        """Wait for services to be healthy"""
        logger.info("Waiting for services to be healthy...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            all_healthy = True
            for service, config in self.services.items():
                if not config["health_check"]():
                    all_healthy = False
                    break
            
            if all_healthy:
                logger.info("✅ All services are healthy")
                return True
            
            time.sleep(5)
        
        logger.error("❌ Services did not become healthy within timeout")
        return False
    
    def check_postgres(self) -> bool:
        """Check PostgreSQL health"""
        try:
            exit_code, stdout, stderr = self.run_command([
                "docker", "compose", "exec", "-T", "postgres",
                "pg_isready", "-U", "telco_user"
            ])
            return exit_code == 0
        except Exception:
            return False
    
    def check_redis(self) -> bool:
        """Check Redis health"""
        try:
            exit_code, stdout, stderr = self.run_command([
                "docker", "compose", "exec", "-T", "redis",
                "redis-cli", "ping"
            ])
            return exit_code == 0 and "PONG" in stdout
        except Exception:
            return False
    
    def check_api(self) -> bool:
        """Check API health"""
        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def check_dashboard(self) -> bool:
        """Check Dashboard health"""
        try:
            response = requests.get("http://localhost:8501/_stcore/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def check_mlflow(self) -> bool:
        """Check MLflow health"""
        try:
            response = requests.get("http://localhost:5000/health", timeout=10)
            return response.status_code == 200 or response.status_code == 404  # MLflow may not have /health
        except Exception:
            return False
    
    def run_api_tests(self) -> bool:
        """Run basic API tests"""
        logger.info("Running API tests...")
        
        # Test API endpoints
        endpoints = [
            {"url": "http://localhost:8000/", "expected_status": 200},
            {"url": "http://localhost:8000/docs", "expected_status": 200},
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint["url"], timeout=10)
                if response.status_code != endpoint["expected_status"]:
                    logger.error(f"❌ API test failed for {endpoint['url']}: {response.status_code}")
                    return False
                logger.info(f"✅ API test passed for {endpoint['url']}")
            except Exception as e:
                logger.error(f"❌ API test error for {endpoint['url']}: {e}")
                return False
        
        return True
    
    def cleanup(self) -> None:
        """Clean up Docker containers and volumes"""
        logger.info("Cleaning up Docker resources...")
        
        # Stop and remove containers
        self.run_command(["docker", "compose", "down", "-v"])
        
        # Remove unused images
        self.run_command(["docker", "image", "prune", "-f"])
        
        logger.info("✅ Cleanup completed")
    
    def run_tests(self) -> bool:
        """Run all tests"""
        logger.info("Starting Docker container tests...")
        
        try:
            # Step 1: Check syntax
            if not self.check_docker_compose_syntax():
                return False
            
            # Step 2: Build images
            if not self.build_images():
                return False
            
            # Step 3: Start services
            if not self.start_services():
                return False
            
            # Step 4: Wait for services to be healthy
            if not self.wait_for_services():
                return False
            
            # Step 5: Run API tests
            if not self.run_api_tests():
                return False
            
            logger.info("✅ All Docker tests passed!")
            return True
            
        except KeyboardInterrupt:
            logger.info("Tests interrupted by user")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during tests: {e}")
            return False
        finally:
            # Always cleanup
            self.cleanup()

def main():
    """Main function"""
    tester = DockerTester()
    
    if tester.run_tests():
        logger.info("🎉 Docker setup is working correctly!")
        sys.exit(0)
    else:
        logger.error("💥 Docker tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
