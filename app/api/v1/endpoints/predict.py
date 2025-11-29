from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import pickle
import csv

router = APIRouter(prefix="/predict", tags=["Predict"])

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
MODELS_DIR = os.path.join(ROOT, 'models')

class PatientFeatures(BaseModel):
    """Patient features for ML prediction."""
    features: List[float]

DISEASES = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']
FEATURE_NAMES = ['age', 'bmi', 'symptom_duration', 'severity', 'progression', 'medication_response', 'exercise_tolerance', 'stress_impact', 'health_score']


@router.post("/ml")
async def predict_all_diseases(patient: PatientFeatures):
    """Predict all diseases using ML models for a single patient."""
    try:
        if len(patient.features) != 9:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected 9 features, got {len(patient.features)}"
            )
        
        predictions = {}
        for disease in DISEASES:
            model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
            
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    model = model_data['model']
                    prediction = model.predict([patient.features])[0]
                    accuracy = model_data.get('accuracy', 'unknown')
                    
                    predictions[disease] = {
                        "prediction": int(prediction),
                        "status": "Present" if prediction == 1 else "Absent",
                        "model_accuracy": accuracy
                    }
                except Exception as e:
                    predictions[disease] = {"error": str(e)}
            else:
                predictions[disease] = {"error": "Model not found"}
        
        return {
            "patient_features": patient.features,
            "predictions": predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def get_models_info():
    """Get information about loaded ML models."""
    try:
        models_info = {}
        
        for disease in DISEASES:
            model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
            models_info[disease] = {
                "available": os.path.exists(model_path),
                "path": model_path
            }
        
        return {"models": models_info, "total_diseases": len(DISEASES)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
