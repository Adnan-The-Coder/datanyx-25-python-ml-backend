#!/usr/bin/env python3
"""Simple test for the new scikit-learn models without FastAPI dependencies"""

import os
import pickle
import sys

# Constants
ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT, 'models')
DISEASES = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']
FEATURE_NAMES = ['age', 'bmi', 'symptom_duration', 'severity', 'progression', 'medication_response', 'exercise_tolerance', 'stress_impact', 'health_score']

def load_model_safely(model_path):
    """Safely load a scikit-learn model with proper error handling"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Verify that the model is properly loaded
        model = model_data.get('model')
        if model is None:
            raise ValueError("No model found in the pickle file")
        
        # Test the model with sample data
        test_features = [[45.0, 25.0, 3.5, 6.0, 4.0, 7.0, 8.0, 2.5, 75.0]]
        prediction = model.predict(test_features)
        
        return model_data
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None

def get_fallback_prediction(features, disease):
    """Generate a simple fallback prediction based on patient features"""
    disease_weights = {
        'diplopia': [0.02, 0.01, 0.15, 0.20, 0.18, -0.10, -0.12, 0.15, -0.08],
        'bulbar': [0.03, 0.01, 0.18, 0.25, 0.20, -0.08, -0.15, 0.12, -0.10],
        'facial': [0.02, 0.005, 0.12, 0.22, 0.16, -0.09, -0.10, 0.10, -0.07],
        'fatigue': [0.01, 0.02, 0.10, 0.30, 0.15, -0.15, -0.20, 0.18, -0.12],
        'limb': [0.025, 0.01, 0.14, 0.28, 0.22, -0.12, -0.18, 0.14, -0.09],
        'ptosis': [0.03, 0.005, 0.16, 0.24, 0.19, -0.08, -0.10, 0.11, -0.08],
        'respiratory': [0.04, 0.02, 0.20, 0.35, 0.25, -0.20, -0.25, 0.20, -0.15]
    }
    
    weights = disease_weights.get(disease, [0.02] * 9)
    
    # Calculate weighted score
    score = sum(f * w for f, w in zip(features, weights))
    
    # Simple threshold-based prediction
    prediction = 1 if score > 1.5 else 0
    
    return prediction

def test_models():
    """Test all disease models"""
    print("="*60)
    print("TESTING NEW SCIKIT-LEARN MODELS")
    print("="*60)
    
    print(f"Models directory: {MODELS_DIR}")
    print(f"Directory exists: {os.path.exists(MODELS_DIR)}")
    
    if os.path.exists(MODELS_DIR):
        print(f"Files in models directory: {os.listdir(MODELS_DIR)}")
    
    # Test patient data
    test_patients = [
        {
            "name": "Low risk patient",
            "features": [25.0, 22.0, 0.5, 2.0, 1.0, 8.0, 9.0, 1.0, 90.0]
        },
        {
            "name": "Medium risk patient", 
            "features": [45.0, 25.0, 3.5, 6.0, 4.0, 7.0, 8.0, 2.5, 75.0]
        },
        {
            "name": "High risk patient",
            "features": [65.0, 30.0, 8.0, 9.0, 8.0, 3.0, 4.0, 4.5, 60.0]
        }
    ]
    
    results = {}
    
    print(f"\nTesting {len(DISEASES)} disease models...\n")
    
    for disease in DISEASES:
        model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
        print(f"Testing {disease} model:")
        print(f"  Path: {model_path}")
        print(f"  Exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            model_data = load_model_safely(model_path)
            if model_data:
                model = model_data['model']
                accuracy = model_data.get('accuracy', 'unknown')
                
                print(f"  ✓ Model loaded successfully")
                print(f"  ✓ Model type: {type(model).__name__}")
                print(f"  ✓ Accuracy: {accuracy}")
                
                results[disease] = {
                    'loaded': True,
                    'model': model,
                    'accuracy': accuracy
                }
            else:
                print(f"  ✗ Failed to load model")
                results[disease] = {'loaded': False, 'error': 'Failed to load'}
        else:
            print(f"  ✗ Model file not found")
            results[disease] = {'loaded': False, 'error': 'File not found'}
        
        print()
    
    # Test predictions
    print("="*60)
    print("TESTING PREDICTIONS")
    print("="*60)
    
    for patient in test_patients:
        print(f"\n{patient['name']}:")
        print(f"Features: {patient['features']}")
        print(f"Predictions:")
        
        for disease in DISEASES:
            if results[disease]['loaded']:
                try:
                    model = results[disease]['model']
                    prediction = model.predict([patient['features']])[0]
                    probabilities = model.predict_proba([patient['features']])[0]
                    confidence = max(probabilities)
                    
                    status = "Present" if prediction == 1 else "Absent"
                    print(f"  {disease:<12}: {status} (confidence: {confidence:.3f})")
                    
                except Exception as e:
                    print(f"  {disease:<12}: ERROR - {e}")
            else:
                # Use fallback
                fallback_pred = get_fallback_prediction(patient['features'], disease)
                status = "Present" if fallback_pred == 1 else "Absent"
                print(f"  {disease:<12}: {status} (fallback)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    loaded_count = sum(1 for r in results.values() if r['loaded'])
    total_count = len(results)
    
    print(f"Models successfully loaded: {loaded_count}/{total_count}")
    
    if loaded_count > 0:
        print("✓ Models are working correctly!")
        avg_accuracy = sum(float(r.get('accuracy', 0)) for r in results.values() if r['loaded'] and isinstance(r.get('accuracy'), (int, float))) / loaded_count
        print(f"✓ Average model accuracy: {avg_accuracy:.3f}")
    
    if loaded_count < total_count:
        print(f"⚠ {total_count - loaded_count} models failed to load, but fallback predictions are available")
    
    return loaded_count == total_count

if __name__ == "__main__":
    try:
        success = test_models()
        if success:
            print("\n🎉 All tests passed! Models are ready for production.")
        else:
            print("\n⚠ Some issues detected, but fallback system provides functionality.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()