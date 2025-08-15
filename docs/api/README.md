# API Documentation

## 🚀 Quick Start

### Development (Local Testing)
```bash
# Fast performance on Windows - localhost only
python scripts/start_api.py

# Or manually with uvicorn
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Production Deployment
```bash
# Set environment first
export ENV=production
export API_WORKERS=4

# Start with external access
python scripts/start_api.py

# Or manually
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 🔧 Environment Configuration

| Variable | Development | Production | Description |
|----------|-------------|------------|-------------|
| `ENV` | `development` | `production` | Environment mode |
| `API_HOST` | `127.0.0.1` | `0.0.0.0` | Host binding |
| `API_PORT` | `8000` | `8000` | Port number |
| `API_WORKERS` | `1` | `4` | Worker processes |

## 📊 Performance Benchmarks

| Environment | Host | Response Time | Use Case |
|-------------|------|---------------|----------|
| Development | `127.0.0.1` | **~3ms** ✅ | Local testing |
| Production | `0.0.0.0` | **~50ms** ✅ | External access |
| Windows Local | `localhost` | **2000ms** ❌ | Avoid this |

## 🏗️ Architecture Notes

### Host Binding Strategy

- **Development (`127.0.0.1`)**: 
  - ✅ Optimal Windows performance (no DNS lookup)
  - ✅ Secure (localhost only)
  - ❌ No external access

- **Production (`0.0.0.0`)**:
  - ✅ External access for load balancers
  - ✅ Container/Docker compatibility
  - ⚠️ Slower on Windows (mitigated by load balancer)

## 🐳 Container Deployment

```bash
# Docker environment
ENV=docker python scripts/start_api.py
```

```dockerfile
# Dockerfile example
ENV ENV=docker
EXPOSE 8000
CMD ["python", "scripts/start_api.py"]
```

## 📖 API Reference

- **Interactive Docs**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## 🔍 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (ultra-fast) |
| `/health/detailed` | GET | Detailed health with validation |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch predictions (up to 1000) |
| `/model/metrics` | GET | Model performance metrics |
| `/model/reload` | POST | Reload model without restart |
