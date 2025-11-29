# Multi-stage Dockerfile for ML Disease Prediction API
FROM python:3.10-slim as base

<<<<<<< HEAD
# Force the server to run on port 8000 instead of default 80
ENV PORT=8000

# The base image already sets WORKDIR to /app
=======
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
>>>>>>> 278679b2ba1cfe5af97631b7b7c4a9030f264a7d

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

<<<<<<< HEAD
# Copy the data directory
COPY ./data /app/data

# Copy the FastAPI app directory
COPY ./app /app/app

# Tell Docker the container listens on port 8000
EXPOSE 8000
=======
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure models directory exists and copy models
RUN mkdir -p models
COPY models/ models/

# Create directories for logs and temp files
RUN mkdir -p logs tmp

# Change ownership to app user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
>>>>>>> 278679b2ba1cfe5af97631b7b7c4a9030f264a7d
