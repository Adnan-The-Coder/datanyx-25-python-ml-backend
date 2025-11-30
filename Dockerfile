# Multi-stage Dockerfile for ML Disease Prediction API
FROM python:3.10-slim as base

# Force the server to run on port 8000 instead of default 80
ENV PORT=8000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and clean up after installation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and other necessary directories
COPY ./app /app/app
COPY ./data /app/data
COPY models/ /app/models/

# Ensure necessary directories exist for logs and temp files
RUN mkdir -p /app/logs /app/tmp

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user for security
USER appuser

# Expose port 8000
EXPOSE 8000

# Health check (ensure FastAPI is running)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application (using uvicorn with worker setup)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
