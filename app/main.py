from fastapi import FastAPI
from datetime import datetime
from api.v1.endpoints import predict

# Initialize the FastAPI application
app = FastAPI(
    title="Minimal FastAPI Backend",
    version="1.0.0",
    description="A simple, Docker-ready starting template for Python microservices."
)

# --- API Endpoints ---

@app.get("/", tags=["Root"])
def read_root():
    """
    Root endpoint for a simple greeting.
    """
    return {
        "message": "Welcome to the minimal FastAPI backend template!",
        "documentation_url": "/docs"
    }


@app.get("/health", tags=["Health Check"])
def health_check():
    """
    Simple health check endpoint to confirm the service is running.
    """
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "api_name": app.title
    }


# include prediction router
app.include_router(predict.router)