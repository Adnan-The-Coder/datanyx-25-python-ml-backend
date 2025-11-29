from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
import pickle
import csv

router = APIRouter(prefix="/predict", tags=["Predict"])

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'primary_dataset.csv')
MODELS_DIR = os.path.join(ROOT, 'models')


class PatientFeatures(BaseModel):
    age: float
    bmi: float
    diplopia_severity: float
    bulbar_score: float
    facial_weakness: float
    fatigue_level: float
    limb_strength: float
    ptosis_degree: float
    resp_function: float


DISEASES = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']


class PatientFeatures(BaseModel):
    age: float
    bmi: float
    diplopia_severity: float
    bulbar_score: float
    facial_weakness: float
    fatigue_level: float
    limb_strength: float
    ptosis_degree: float
    resp_function: float


DISEASES = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']


@router.get("/model/{symptom}/{patient_id}")
def predict_single(symptom: str, patient_id: str):
    # Read compact dataset
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    
    # Find patient
    patient_row = None
    for row in rows:
        if row.get('primary_id') == patient_id:
            patient_row = row
            break
    
    if patient_row is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Return existing prediction from compact dataset
    if symptom in patient_row:
        pred = int(patient_row[symptom])
        return {"patient_id": patient_id, "symptom": symptom, "pred": pred, "source": "dataset"}
    else:
        raise HTTPException(status_code=404, detail="Symptom not found")


@router.post("/run_all")
def run_all_and_update():
    # The compact dataset already contains the final predictions
    # This endpoint could re-run the compact generation script if needed
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    
    return {"status": "ok", "message": "Dataset already contains ML predictions", "patients": len(rows)}


@router.get("/all")
def get_all_predictions():
    """Get all patient predictions in compact format."""
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    
    return {"patients": rows, "count": len(rows)}


@router.get("/patient/{patient_id}")
def get_patient_all_symptoms(patient_id: str):
    """Get all symptom predictions for a specific patient."""
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    
    # Find patient
    for row in rows:
        if row.get('primary_id') == patient_id:
            # Convert symptom values to int
            symptoms = {}
            for key, value in row.items():
                if key != 'primary_id':
                    symptoms[key] = int(value)
            return {"patient_id": patient_id, "symptoms": symptoms}
    
    raise HTTPException(status_code=404, detail="Patient not found")


@router.post("/ml_predict")
def predict_diseases_ml(features: PatientFeatures):
    """Predict disease presence using trained ML models based on patient features."""
    feature_vector = [
        features.age,
        features.bmi,
        features.diplopia_severity,
        features.bulbar_score,
        features.facial_weakness,
        features.fatigue_level,
        features.limb_strength,
        features.ptosis_degree,
        features.resp_function
    ]
    
    predictions = {}
    model_info = {}
    
    for disease in DISEASES:
        model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
        try:
            with open(model_path, 'rb') as f:
                obj = pickle.load(f)
            model = obj['model']
            accuracy = obj.get('accuracy', 'unknown')
            pred = model.predict([feature_vector])[0]
            predictions[disease] = int(pred)
            model_info[disease] = {'accuracy': accuracy, 'source': 'ml_model'}
        except Exception as e:
            # Fallback to rule-based prediction
            predictions[disease] = 0  # Default to no disease
            model_info[disease] = {'error': str(e), 'source': 'fallback'}
    
    return {
        "predictions": predictions,
        "model_info": model_info,
        "features_used": {
            "age": features.age,
            "bmi": features.bmi,
            "diplopia_severity": features.diplopia_severity,
            "bulbar_score": features.bulbar_score,
            "facial_weakness": features.facial_weakness,
            "fatigue_level": features.fatigue_level,
            "limb_strength": features.limb_strength,
            "ptosis_degree": features.ptosis_degree,
            "resp_function": features.resp_function
        }
    }


@router.get("/ml_models_info")
def get_ml_models_info():
    """Get information about trained ML models."""
    models_info = {}
    
    for disease in DISEASES:
        model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    obj = pickle.load(f)
                models_info[disease] = {
                    'features': obj.get('features', []),
                    'accuracy': obj.get('accuracy', 'unknown'),
                    'available': True
                }
            except Exception as e:
                models_info[disease] = {
                    'error': str(e),
                    'available': False
                }
        else:
            models_info[disease] = {
                'error': 'Model file not found',
                'available': False
            }
    
    return {"models": models_info}
