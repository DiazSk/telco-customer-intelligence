#!/usr/bin/env python
"""
Docker Launch Script for Telco Customer Intelligence Platform
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

class DockerLauncher:
    def __init__(self):
        self.services = {
            'postgres': 'http://localhost:5432',
            'redis': 'http://localhost:6379', 
            'api': 'http://localhost:8000/health',
            'dashboard': 'http://localhost:8501/_stcore/health',
            'mlflow': 'http://localhost:5000'
        }
        
    def print_banner(self):
        """Print Docker deployment banner"""
        print("""
        ╔════════════════════════════════════════════════════════════╗
        ║                                                            ║
        ║     🐳 TELCO INTELLIGENCE PLATFORM - DOCKER DEPLOY 🐳     ║
        ║                                                            ║
        ║     Containerized deployment with full isolation          ║
        ║                                                            ║
        ╚════════════════════════════════════════════════════════════╝
        """)
        
    def check_docker(self):
        """Check if Docker is available"""
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Docker available: {result.stdout.strip()}")
                return True
            else:
                print("❌ Docker not found")
                return False
        except FileNotFoundError:
            print("❌ Docker not installed")
            print("   Please install Docker Desktop: https://www.docker.com/products/docker-desktop")
            return False
            
    def check_docker_compose(self):
        """Check if Docker Compose is available"""
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Docker Compose available: {result.stdout.strip()}")
                return True
            else:
                print("❌ Docker Compose not found")
                return False
        except FileNotFoundError:
            print("❌ Docker Compose not installed")
            return False
            
    def build_images(self):
        """Build Docker images"""
        print("\n🔨 Building Docker images...")
        
        try:
            # Build main application image
            print("   Building main application...")
            subprocess.run(['docker-compose', 'build'], check=True)
            
            print("✅ Images built successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Build failed: {e}")
            return False
            
    def start_services(self, services=None):
        """Start Docker services"""
        if services is None:
            services = ['postgres', 'redis', 'api', 'dashboard']
            
        print(f"\n🚀 Starting services: {', '.join(services)}")
        
        try:
            subprocess.run(['docker-compose', 'up', '-d'] + services, check=True)
            print("✅ Services started")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start services: {e}")
            return False
            
    def wait_for_services(self):
        """Wait for services to be healthy"""
        print("\n⏳ Waiting for services to be ready...")
        
        for service, url in self.services.items():
            if service in ['postgres', 'redis']:
                continue  # Skip direct health checks for databases
                
            print(f"   Checking {service}...")
            
            for attempt in range(30):  # 30 attempts, 10 seconds each = 5 minutes max
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        print(f"   ✅ {service} is ready")
                        break
                except requests.exceptions.RequestException:
                    pass
                    
                time.sleep(10)
            else:
                print(f"   ⚠️  {service} not responding (may still be starting)")
                
    def show_access_points(self):
        """Display access points"""
        print("""
============================================================
✅ DOCKER DEPLOYMENT COMPLETE!
============================================================

        📍 Access Points:
        • Dashboard: http://localhost:8501
        • API: http://localhost:8000
        • API Docs: http://localhost:8000/docs
        • MLflow: http://localhost:5000
        • Jupyter: http://localhost:8888

        🐳 Docker Commands:
        • View logs: docker-compose logs -f
        • Stop services: docker-compose down
        • Restart: docker-compose restart
        • Scale API: docker-compose up -d --scale api=3

        📊 Monitor with:
        docker-compose ps
        docker-compose top
        """)
        
    def deploy(self):
        """Full deployment process"""
        self.print_banner()
        
        # Pre-flight checks
        if not self.check_docker():
            return False
            
        if not self.check_docker_compose():
            return False
            
        # Build and deploy
        if not self.build_images():
            return False
            
        if not self.start_services():
            return False
            
        self.wait_for_services()
        self.show_access_points()
        
        return True

def main():
    """Main deployment function"""
    launcher = DockerLauncher()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'build':
            launcher.print_banner()
            launcher.build_images()
            
        elif command == 'start':
            launcher.print_banner()
            services = sys.argv[2:] if len(sys.argv) > 2 else None
            launcher.start_services(services)
            
        elif command == 'stop':
            print("🛑 Stopping all services...")
            subprocess.run(['docker-compose', 'down'])
            
        elif command == 'logs':
            subprocess.run(['docker-compose', 'logs', '-f'])
            
        elif command == 'status':
            subprocess.run(['docker-compose', 'ps'])
            
        else:
            print(f"Unknown command: {command}")
            print("Usage: python scripts/docker_launch.py [build|start|stop|logs|status]")
    else:
        # Full deployment
        success = launcher.deploy()
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()
