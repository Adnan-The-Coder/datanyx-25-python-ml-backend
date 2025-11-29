# ML Disease Prediction API Documentation

## Overview
A robust FastAPI-based machine learning API for predicting myasthenia gravis symptoms. This containerized service provides real-time predictions for 7 different disease symptoms using trained Random Forest models.

## Base URL
```
http://localhost:8000
```

## Supported Diseases
- **Diplopia**: Double vision
- **Bulbar**: Speech/swallowing difficulties  
- **Facial**: Facial muscle weakness
- **Fatigue**: General fatigue levels
- **Limb**: Limb muscle weakness
- **Ptosis**: Eyelid drooping
- **Respiratory**: Breathing difficulties

## API Endpoints

### 1. Root Endpoint
**GET /** 
- **Description**: API information and overview
- **Response**: JSON with API details and available endpoints

```bash
curl http://localhost:8000/
```

**Response Example:**
```json
{
  "message": "ML Disease Prediction API",
  "version": "1.0.0",
  "diseases": ["diplopia", "bulbar", "facial", "fatigue", "limb", "ptosis", "respiratory"],
  "endpoints": {
    "/docs": "Interactive API documentation",
    "/health": "Health check endpoint",
    "/predict/": "Prediction endpoints"
  }
}
```

### 2. Health Check
**GET /health**
- **Description**: Service health and model availability check
- **Response**: Health status and model information

```bash
curl http://localhost:8000/health
```

**Response Example:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-30T10:30:00",
  "api_name": "ML Disease Prediction API",
  "models_loaded": true
}
```

### 3. API Information
**GET /info**
- **Description**: Detailed API and model status information
- **Response**: Comprehensive API details

```bash
curl http://localhost:8000/info
```

### 4. ML Prediction (Single Patient)
**POST /predict/ml**
- **Description**: Predict all diseases for a single patient
- **Content-Type**: application/json
- **Request Body**: Patient features (9 numeric values)

```bash
curl -X POST http://localhost:8000/predict/ml \
  -H "Content-Type: application/json" \
  -d '{
    "features": [45.0, 25.0, 7.5, 6.0, 5.0, 8.0, 3.0, 3.5, 60.0]
  }'
```

**Request Body Schema:**
```json
{
  "features": [
    "age (float): Patient age in years",
    "bmi (float): Body mass index", 
    "symptom_duration (float): Duration of symptoms in years",
    "severity (float): Symptom severity scale (0-10)",
    "progression (float): Disease progression rate",
    "medication_response (float): Response to medication (1-10)",
    "exercise_tolerance (float): Exercise tolerance level (1-10)", 
    "stress_impact (float): Stress impact on symptoms (0-5)",
    "health_score (float): Overall health score (0-100)"
  ]
}
```

**Response Example:**
```json
{
  "patient_features": [45.0, 25.0, 7.5, 6.0, 5.0, 8.0, 3.0, 3.5, 60.0],
  "predictions": {
    "diplopia": {
      "prediction": 1,
      "status": "Present",
      "confidence": 0.95
    },
    "bulbar": {
      "prediction": 0,
      "status": "Absent",
      "confidence": 0.87
    },
    "facial": {
      "prediction": 1,
      "status": "Present", 
      "confidence": 0.92
    }
  }
}
```

### 5. Model Information
**GET /predict/models**
- **Description**: Information about loaded ML models
- **Response**: Model details and performance metrics

```bash
curl http://localhost:8000/predict/models
```

### 6. Batch Prediction
**POST /predict/batch**
- **Description**: Predict diseases for multiple patients
- **Content-Type**: application/json

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {"features": [45.0, 25.0, 7.5, 6.0, 5.0, 8.0, 3.0, 3.5, 60.0]},
      {"features": [50.0, 28.0, 5.2, 4.0, 3.0, 6.0, 5.0, 2.0, 75.0]}
    ]
  }'
```

### 7. Individual Disease Prediction
**POST /predict/{disease}**
- **Description**: Predict specific disease for a patient
- **Path Parameter**: disease (diplopia|bulbar|facial|fatigue|limb|ptosis|respiratory)

```bash
curl -X POST http://localhost:8000/predict/diplopia \
  -H "Content-Type: application/json" \
  -d '{"features": [45.0, 25.0, 7.5, 6.0, 5.0, 8.0, 3.0, 3.5, 60.0]}'
```

## Interactive Documentation

Once the API is running, access interactive documentation at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Docker Usage

### Build the Container
```bash
# Build the Docker image
docker build -t disease-prediction-api .

# Run the container
docker run -p 8000:8000 disease-prediction-api
```

### Docker Compose (Optional)
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Error Responses

All endpoints return appropriate HTTP status codes:
- **200**: Success
- **400**: Bad Request (invalid input)
- **404**: Not Found (endpoint/resource not found)
- **422**: Validation Error (invalid request body)
- **500**: Internal Server Error

**Error Response Format:**
```json
{
  "detail": "Error description"
}
```

## Model Features

### Input Features (9 required values):
1. **Age**: Patient age in years (20-80)
2. **BMI**: Body mass index (18-35)  
3. **Symptom Duration**: Duration in years (0.1-10)
4. **Severity**: Symptom severity scale (0.1-10)
5. **Progression**: Disease progression rate (0.1-10)
6. **Medication Response**: Response scale (1-10)
7. **Exercise Tolerance**: Tolerance level (1-10)
8. **Stress Impact**: Stress impact scale (0.1-5)
9. **Health Score**: Overall health (50-100)

### Output Predictions:
- **Binary Classification**: 0 (Absent) or 1 (Present)
- **Status**: Human-readable "Present" or "Absent"
- **Confidence**: Model confidence score (when available)

## Security Considerations

- API runs on non-root user in container
- CORS configured for cross-origin requests
- Input validation on all endpoints
- Health checks for monitoring
- Structured error handling

## Monitoring and Logging

- Health check endpoint for load balancers
- Structured JSON responses
- Request/response logging
- Container health checks
- Startup/shutdown event logging

## Performance

- Lightweight Random Forest models
- Fast inference times (<100ms)
- Stateless design for horizontal scaling
- Efficient model loading and caching
- Memory-optimized containers

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run in development mode
uvicorn app.main:app --reload --port 8000
```

### Testing
```bash
# Test ML models
python test_ml_models.py

# Test API endpoints
python test_api_endpoints.py
```

## Support

For issues or questions:
- Check the interactive documentation at `/docs`
- Review health status at `/health`  
- Examine model status at `/info`
- Contact support team