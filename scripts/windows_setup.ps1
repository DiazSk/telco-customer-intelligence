# Telco Customer Intelligence Platform - PowerShell Setup Script
# Alternative to Makefile for Windows PowerShell environments

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

function Show-Help {
    Write-Host "Telco Customer Intelligence Platform - Available commands:" -ForegroundColor Green
    Write-Host ""
    Write-Host "Development:" -ForegroundColor Yellow
    Write-Host "  install          Install production dependencies"
    Write-Host "  dev-install      Install development dependencies"
    Write-Host "  test            Run all tests"
    Write-Host "  lint            Run linting checks"
    Write-Host "  format          Format code with black and isort"
    Write-Host ""
    Write-Host "Docker:" -ForegroundColor Yellow
    Write-Host "  build           Build Docker images"
    Write-Host "  up              Start development environment"
    Write-Host "  down            Stop development environment"
    Write-Host "  logs            Show container logs"
    Write-Host "  restart         Restart all containers"
    Write-Host "  health          Check container health"
    Write-Host "  test-docker     Test Docker containers"
    Write-Host ""
    Write-Host "Usage: .\scripts\windows_setup.ps1 [command]" -ForegroundColor Cyan
}

function Install-Dependencies {
    Write-Host "Installing production dependencies..." -ForegroundColor Green
    pip install -r requirements.txt
}

function Install-DevDependencies {
    Write-Host "Installing development dependencies..." -ForegroundColor Green
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
}

function Run-Tests {
    Write-Host "Running all tests..." -ForegroundColor Green
    python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
}

function Run-Lint {
    Write-Host "Running linting checks..." -ForegroundColor Green
    flake8 src/ --max-line-length=88 --extend-ignore=E203,W503
    mypy src/ --ignore-missing-imports
}

function Format-Code {
    Write-Host "Formatting code..." -ForegroundColor Green
    black src/ tests/
    isort src/ tests/
}

function Build-Images {
    Write-Host "Building Docker images..." -ForegroundColor Green
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        docker compose build
    } else {
        Write-Host "Docker not found. Please install Docker Desktop." -ForegroundColor Red
        Write-Host "Download from: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
    }
}

function Start-Environment {
    Write-Host "Starting development environment..." -ForegroundColor Green
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        docker compose up -d
        Write-Host ""
        Write-Host "Services starting up..." -ForegroundColor Yellow
        Write-Host "API: http://localhost:8000" -ForegroundColor Cyan
        Write-Host "Dashboard: http://localhost:8501" -ForegroundColor Cyan
        Write-Host "MLflow: http://localhost:5000" -ForegroundColor Cyan
        Write-Host "Use 'health' command to check service status" -ForegroundColor Yellow
    } else {
        Write-Host "Docker not found. Please install Docker Desktop." -ForegroundColor Red
    }
}

function Stop-Environment {
    Write-Host "Stopping development environment..." -ForegroundColor Green
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        docker compose down
    } else {
        Write-Host "Docker not found." -ForegroundColor Red
    }
}

function Show-Logs {
    Write-Host "Showing container logs..." -ForegroundColor Green
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        docker compose logs -f
    } else {
        Write-Host "Docker not found." -ForegroundColor Red
    }
}

function Restart-Containers {
    Write-Host "Restarting all containers..." -ForegroundColor Green
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        docker compose restart
    } else {
        Write-Host "Docker not found." -ForegroundColor Red
    }
}

function Check-Health {
    Write-Host "Checking container health..." -ForegroundColor Green
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        docker compose ps
        Write-Host ""
        Write-Host "Checking service health..." -ForegroundColor Yellow
        
        try {
            $apiResponse = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -UseBasicParsing
            Write-Host "✅ API: Healthy" -ForegroundColor Green
        } catch {
            Write-Host "❌ API: Not ready" -ForegroundColor Red
        }
        
        try {
            $dashboardResponse = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -TimeoutSec 5 -UseBasicParsing
            Write-Host "✅ Dashboard: Healthy" -ForegroundColor Green
        } catch {
            Write-Host "❌ Dashboard: Not ready" -ForegroundColor Red
        }
        
        try {
            $mlflowResponse = Invoke-WebRequest -Uri "http://localhost:5000" -TimeoutSec 5 -UseBasicParsing
            Write-Host "✅ MLflow: Healthy" -ForegroundColor Green
        } catch {
            Write-Host "❌ MLflow: Not ready" -ForegroundColor Red
        }
    } else {
        Write-Host "Docker not found." -ForegroundColor Red
    }
}

function Test-Docker {
    Write-Host "Testing Docker containers..." -ForegroundColor Green
    if (Get-Command python -ErrorAction SilentlyContinue) {
        python scripts/test_docker.py
    } else {
        Write-Host "Python not found. Please install Python." -ForegroundColor Red
    }
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "install" { Install-Dependencies }
    "dev-install" { Install-DevDependencies }
    "test" { Run-Tests }
    "lint" { Run-Lint }
    "format" { Format-Code }
    "build" { Build-Images }
    "up" { Start-Environment }
    "down" { Stop-Environment }
    "logs" { Show-Logs }
    "restart" { Restart-Containers }
    "health" { Check-Health }
    "test-docker" { Test-Docker }
    default { 
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Show-Help 
    }
}
