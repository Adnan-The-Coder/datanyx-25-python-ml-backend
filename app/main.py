<<<<<<< HEAD
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from .api.v1.endpoints import predict  # <-- Changed this line
=======
"""
FastAPI ML Disease Prediction API

This application provides machine learning endpoints for predicting 
myasthenia gravis symptoms including:
- Diplopia (double vision)
- Bulbar symptoms 
- Facial weakness
- Fatigue
- Limb weakness  
- Ptosis (eyelid drooping)
- Respiratory difficulties
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import uvicorn
import os
import sys

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import API routers
from api.v1.endpoints import predict
>>>>>>> 278679b2ba1cfe5af97631b7b7c4a9030f264a7d

# Initialize the FastAPI application
app = FastAPI(
    title="ML Disease Prediction API",
    version="1.0.0",
    description="""
    A robust machine learning API for predicting myasthenia gravis symptoms.
    
    ## Features
    - Individual disease prediction endpoints
    - Batch prediction for all diseases
    - Model information and health checks
    - Real-time ML inference
    
    ## Diseases Supported
    - **Diplopia**: Double vision prediction
    - **Bulbar**: Speech/swallowing difficulties  
    - **Facial**: Facial muscle weakness
    - **Fatigue**: General fatigue levels
    - **Limb**: Limb muscle weakness
    - **Ptosis**: Eyelid drooping
    - **Respiratory**: Breathing difficulties
    """,
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

<<<<<<< HEAD
# Add CORS middleware - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# --- API Endpoints ---
=======
# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# --- Core API Endpoints ---
>>>>>>> 278679b2ba1cfe5af97631b7b7c4a9030f264a7d

@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint providing API information and available endpoints.
    """
    return {
        "message": "ML Disease Prediction API",
        "version": "1.0.0",
        "description": "Machine learning API for myasthenia gravis symptom prediction",
        "endpoints": {
            "/docs": "Interactive API documentation",
            "/redoc": "Alternative API documentation",
            "/health": "Health check endpoint",
            "/predict/": "Prediction endpoints",
        },
        "diseases": ["diplopia", "bulbar", "facial", "fatigue", "limb", "ptosis", "respiratory"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancer probes.
    """
    try:
        # Check if models directory exists
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        models_available = os.path.exists(models_dir) and len(os.listdir(models_dir)) > 0
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "api_name": app.title,
            "version": "1.0.0",
            "models_loaded": models_available,
            "uptime": "Available"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/info", tags=["Information"])
async def api_info():
    """
    Detailed API information including model status and capabilities.
    """
    try:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        diseases = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']
        
        model_status = {}
        for disease in diseases:
            model_path = os.path.join(models_dir, f"{disease}_ml_model.pkl")
            model_status[disease] = {
                "available": os.path.exists(model_path),
                "path": model_path
            }
        
        return {
            "api_name": app.title,
            "version": "1.0.0",
            "description": app.description,
            "total_diseases": len(diseases),
            "diseases": diseases,
            "model_status": model_status,
            "endpoints": {
                "prediction": "/predict/ml",
                "batch_prediction": "/predict/batch",
                "model_info": "/predict/models",
                "health": "/health"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get API info: {str(e)}")

# Include prediction router
app.include_router(predict.router)

# --- Application startup/shutdown events ---

@app.on_event("startup")
async def startup_event():
    """
    Application startup tasks.
    """
    print("🚀 ML Disease Prediction API starting up...")
    print(f"📊 API Documentation available at: http://localhost:8000/docs")
    print(f"🔍 Health check available at: http://localhost:8000/health")
    
@app.on_event("shutdown") 
async def shutdown_event():
    """
    Application shutdown tasks.
    """
    print("🛑 ML Disease Prediction API shutting down...")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
