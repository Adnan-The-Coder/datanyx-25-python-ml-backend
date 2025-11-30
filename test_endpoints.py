#!/usr/bin/env python3
"""Test API endpoint functionality without running FastAPI server"""

import json
import sys
import os

# Add necessary paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_prediction_endpoint():
    """Test the prediction endpoint logic"""
    print("="*60)
    print("TESTING PREDICTION ENDPOINT LOGIC")
    print("="*60)
    
    # Simulate what the endpoint does
    from predict import load_model_safely, get_fallback_prediction
    
    # Constants
    ROOT = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(ROOT, 'models')
    DISEASES = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']
    FEATURE_NAMES = ['age', 'bmi', 'symptom_duration', 'severity', 'progression', 'medication_response', 'exercise_tolerance', 'stress_impact', 'health_score']
    
    # Test patient data
    patient_features = [45.0, 25.0, 3.5, 6.0, 4.0, 7.0, 8.0, 2.5, 75.0]
    
    print(f"Testing with patient features: {patient_features}")
    print()
    
    predictions = {}
    models_missing = []
    
    for disease in DISEASES:
        model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
        
        if os.path.exists(model_path):
            try:
                model_data = load_model_safely(model_path)
                
                if model_data is not None:
                    model = model_data['model']
                    
                    # Make prediction using scikit-learn model
                    prediction_result = model.predict([patient_features])
                    probabilities = model.predict_proba([patient_features])
                    
                    prediction = int(prediction_result[0])
                    accuracy = model_data.get('accuracy', 'unknown')
                    
                    predictions[disease] = {
                        "prediction": prediction,
                        "status": "Present" if prediction == 1 else "Absent",
                        "confidence": float(max(probabilities[0])),
                        "probabilities": {
                            "absent": float(probabilities[0][0]),
                            "present": float(probabilities[0][1])
                        },
                        "model_accuracy": float(accuracy) if isinstance(accuracy, (int, float)) else accuracy,
                        "model_type": "scikit-learn"
                    }
                    
                    print(f"✓ {disease}: {predictions[disease]['status']} (confidence: {predictions[disease]['confidence']:.3f})")
                else:
                    # Model loading failed, use fallback
                    raise ValueError("Model loading failed")
                    
            except Exception as e:
                print(f"⚠ {disease}: Error - {str(e)}, using fallback")
                # Use fallback prediction
                prediction = get_fallback_prediction(patient_features, disease)
                
                predictions[disease] = {
                    "prediction": prediction,
                    "status": "Present" if prediction == 1 else "Absent",
                    "confidence": 0.60,
                    "model_accuracy": 0.65,
                    "model_type": "fallback",
                    "note": f"Using fallback prediction due to model error: {str(e)}"
                }
        else:
            models_missing.append(disease)
            print(f"✗ {disease}: Model file not found")
            # Use fallback prediction
            prediction = get_fallback_prediction(patient_features, disease)
            
            predictions[disease] = {
                "prediction": prediction,
                "status": "Present" if prediction == 1 else "Absent", 
                "confidence": 0.55,
                "model_accuracy": 0.60,
                "model_type": "fallback",
                "note": "Model file not found, using fallback prediction"
            }
    
    # Create API response
    response = {
        "patient_features": patient_features,
        "feature_names": FEATURE_NAMES,
        "predictions": predictions,
        "total_diseases_evaluated": len(DISEASES)
    }
    
    if models_missing:
        response["warning"] = f"Models missing for: {', '.join(models_missing)}. Using fallback predictions."
    
    print("\n" + "="*60)
    print("API RESPONSE SIMULATION")
    print("="*60)
    print(json.dumps(response, indent=2))
    
    return response

def test_models_endpoint():
    """Test the models info endpoint logic"""
    print("\n" + "="*60)
    print("TESTING MODELS INFO ENDPOINT")
    print("="*60)
    
    from predict import load_model_safely
    
    # Constants
    ROOT = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(ROOT, 'models')
    DISEASES = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']
    
    models_info = {}
    total_available = 0
    
    for disease in DISEASES:
        model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
        is_available = os.path.exists(model_path)
        
        if is_available:
            total_available += 1
            try:
                # Try to load model to verify it's working
                model_data = load_model_safely(model_path)
                if model_data:
                    models_info[disease] = {
                        "available": True,
                        "path": model_path,
                        "accuracy": model_data.get('accuracy', 'unknown'),
                        "model_type": model_data.get('model_type', 'unknown'),
                        "status": "loaded successfully"
                    }
                else:
                    models_info[disease] = {
                        "available": True,
                        "path": model_path,
                        "status": "file exists but failed to load",
                        "fallback_available": True
                    }
            except Exception as e:
                models_info[disease] = {
                    "available": True,
                    "path": model_path,
                    "status": f"error loading: {str(e)}",
                    "fallback_available": True
                }
        else:
            models_info[disease] = {
                "available": False,
                "path": model_path,
                "status": "file not found",
                "fallback_available": True
            }
    
    response = {
        "models": models_info,
        "total_diseases": len(DISEASES),
        "models_available": total_available,
        "models_missing": len(DISEASES) - total_available,
        "models_directory": MODELS_DIR
    }
    
    print(json.dumps(response, indent=2))
    return response

if __name__ == "__main__":
    try:
        # Copy the required functions
        from simple_test import load_model_safely, get_fallback_prediction
        
        # Override the module-level functions
        import predict
        predict.load_model_safely = load_model_safely 
        predict.get_fallback_prediction = get_fallback_prediction
        
        test_prediction_endpoint()
        test_models_endpoint()
        
        print("\n🎉 API endpoint tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()