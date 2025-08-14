# Multi-stage build for production-ready image
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/mluser/.local

# Copy application code
COPY --chown=mluser:mluser . .

# Make sure scripts are in PATH
ENV PATH=/home/mluser/.local/bin:$PATH

# Switch to non-root user
USER mluser

# Create necessary directories
RUN mkdir -p data/{raw,processed,features} logs models mlruns

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden)
CMD ["python", "src/api/main.py"]
