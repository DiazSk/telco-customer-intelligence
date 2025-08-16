# Windows Setup Guide - Telco Customer Intelligence Platform

This guide will help you set up and run the Telco Customer Intelligence Platform on Windows.

## Prerequisites

### 1. Install Docker Desktop
1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Run the installer and follow the setup wizard
3. Restart your computer when prompted
4. Start Docker Desktop and wait for it to initialize

### 2. Install Python (if not already installed)
1. Download Python 3.9+ from: https://www.python.org/downloads/
2. During installation, check "Add Python to PATH"
3. Verify installation: `python --version`

### 3. Install Git (if not already installed)
1. Download Git from: https://git-scm.com/download/win
2. Use default installation settings

## Quick Start

### Option 1: Using PowerShell (Recommended)

Open PowerShell as Administrator and navigate to the project directory:

```powershell
# Navigate to project directory
cd C:\telco-customer-intelligence

# Install Python dependencies
.\scripts\windows_setup.ps1 dev-install

# Start the Docker environment
.\scripts\windows_setup.ps1 up

# Check service health
.\scripts\windows_setup.ps1 health
```

### Option 2: Using Command Prompt

Open Command Prompt as Administrator:

```cmd
# Navigate to project directory
cd C:\telco-customer-intelligence

# Install Python dependencies
scripts\windows_setup.bat dev-install

# Start the Docker environment
scripts\windows_setup.bat up

# Check service health
scripts\windows_setup.bat health
```

### Option 3: Direct Docker Commands

If you prefer to use Docker commands directly:

```powershell
# Build and start services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f

# Stop services
docker compose down
```

## Available Commands

### PowerShell Script Commands
```powershell
.\scripts\windows_setup.ps1 [command]
```

### Batch Script Commands
```cmd
scripts\windows_setup.bat [command]
```

### Available Commands:

| Command | Description |
|---------|-------------|
| `help` | Show available commands |
| `install` | Install production dependencies |
| `dev-install` | Install development dependencies |
| `test` | Run all tests |
| `lint` | Run linting checks |
| `format` | Format code |
| `build` | Build Docker images |
| `up` | Start development environment |
| `down` | Stop development environment |
| `logs` | Show container logs |
| `restart` | Restart all containers |
| `health` | Check container health |
| `test-docker` | Test Docker containers |

## Service URLs

Once the services are running, you can access:

- **API Documentation**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501
- **MLflow**: http://localhost:5000
- **PostgreSQL**: localhost:5432 (username: telco_user, password: telco_pass)
- **Redis**: localhost:6379

## Troubleshooting

### Docker Issues

1. **Docker not found error**:
   - Make sure Docker Desktop is installed and running
   - Restart Docker Desktop if needed
   - Check if virtualization is enabled in BIOS

2. **Port conflicts**:
   - Stop other services using ports 8000, 8501, 5000, 5432, 6379
   - Use `netstat -ano | findstr :8000` to check port usage

3. **Permission errors**:
   - Run PowerShell/Command Prompt as Administrator
   - Make sure Docker Desktop has necessary permissions

### Python Issues

1. **Python not found**:
   - Install Python from python.org
   - Make sure Python is added to PATH during installation

2. **Module not found errors**:
   - Run the dev-install command to install dependencies
   - Use virtual environment for better isolation

### Service Health Check

To verify all services are working:

```powershell
# Check container status
.\scripts\windows_setup.ps1 health

# Or manually check each service
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

### Environment Variables

Create a `.env` file in the project root with:

```env
DB_USER=telco_user
DB_PASSWORD=telco_pass
DB_NAME=telco_db
SECRET_KEY=your-secret-key-here
ENVIRONMENT=development
```

## Development Workflow

1. **Start development environment**:
   ```powershell
   .\scripts\windows_setup.ps1 up
   ```

2. **Check logs**:
   ```powershell
   .\scripts\windows_setup.ps1 logs
   ```

3. **Run tests**:
   ```powershell
   .\scripts\windows_setup.ps1 test
   ```

4. **Stop environment**:
   ```powershell
   .\scripts\windows_setup.ps1 down
   ```

## Getting Help

- Check the main README.md for detailed documentation
- View API documentation at http://localhost:8000/docs
- Check Docker logs for service-specific issues
- Ensure all prerequisites are properly installed

## Alternative: WSL2 Setup

For a more Linux-like experience, you can use WSL2:

1. Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install
2. Install Ubuntu or your preferred Linux distribution
3. Follow the Linux setup instructions from the main README.md

This approach allows you to use the standard `make` commands as intended.
