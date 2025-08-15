# Docker Deployment Guide

## Overview
This guide covers deploying the Telco Customer Intelligence Platform using Docker containers.

## Prerequisites
- Docker Desktop installed
- Docker Compose v2.0+
- At least 4GB RAM available for containers

## Quick Start

### 1. Build and Start All Services
```bash
# Start the complete platform
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### 2. Access the Platform
- **Dashboard**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Jupyter**: http://localhost:8888

### 3. Individual Services
```bash
# Start only API and dependencies
docker-compose up -d postgres redis api

# Start only dashboard
docker-compose up -d api dashboard

# Stop all services
docker-compose down
```

## Configuration

### Streamlit Configuration
The dashboard uses optimized settings in `.streamlit/config.toml`:
- Custom theme with professional colors
- Performance optimizations
- Security settings

### Environment Variables
Set these in your environment or `.env` file:
```bash
DB_PASSWORD=your_secure_password
API_URL=http://api:8000
```

## Production Deployment

### 1. Environment Setup
```bash
# Copy and customize environment file
cp .env.example .env
# Edit .env with your production values
```

### 2. Security Considerations
- Change default passwords
- Enable HTTPS in production
- Configure firewall rules
- Use secrets management

### 3. Scaling
```bash
# Scale API instances
docker-compose up -d --scale api=3

# Scale dashboard instances
docker-compose up -d --scale dashboard=2
```

## Monitoring

### Health Checks
All services include health checks:
```bash
# Check service health
docker-compose exec dashboard curl http://localhost:8501/_stcore/health
docker-compose exec api curl http://localhost:8000/health
```

### Logs
```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f dashboard
docker-compose logs -f api
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check for port usage
   netstat -an | findstr :8501
   ```

2. **Memory Issues**
   ```bash
   # Check Docker memory usage
   docker stats
   ```

3. **Build Failures**
   ```bash
   # Clean build cache
   docker-compose build --no-cache
   ```

### Debugging
```bash
# Access container shell
docker-compose exec dashboard bash
docker-compose exec api bash

# View container resources
docker-compose top
```

## Performance Optimization

### Dashboard Optimizations
- Streamlit caching enabled
- Reduced file upload limits
- Optimized theme settings
- Health checks for reliability

### API Optimizations
- FastAPI with async support
- Redis caching
- Database connection pooling
- Model caching

## Backup and Recovery

### Data Backup
```bash
# Backup PostgreSQL data
docker-compose exec postgres pg_dump -U telco_user telco_db > backup.sql

# Backup processed data
docker cp telco_customer_intelligence_dashboard_1:/app/data ./data_backup
```

### Recovery
```bash
# Restore database
docker-compose exec postgres psql -U telco_user telco_db < backup.sql
```
