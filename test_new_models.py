#!/usr/bin/env python3
"""Test the updated predict.py with new scikit-learn models"""

import sys
import os
import requests
import json

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import the prediction functions directly
from app.api.v1.endpoints.predict import load_model_safely, get_fallback_prediction, DISEASES, MODELS_DIR

def test_model_loading():
    """Test that all models can be loaded properly"""
    print("Testing model loading...")
    
    for disease in DISEASES:
        model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
        print(f"\nTesting {disease} model:")
        print(f"  Path: {model_path}")
        print(f"  Exists: {os.path.exists(model_path)}")
        
        if os.path.exists(model_path):
            model_data = load_model_safely(model_path)
            if model_data:
                model = model_data['model']
                accuracy = model_data.get('accuracy', 'unknown')
                print(f"  Model loaded successfully")
                print(f"  Accuracy: {accuracy}")
                print(f"  Model type: {type(model).__name__}")
                
                # Test prediction
                test_features = [45.0, 25.0, 3.5, 6.0, 4.0, 7.0, 8.0, 2.5, 75.0]
                try:
                    prediction = model.predict([test_features])
                    probabilities = model.predict_proba([test_features])
                    print(f"  Test prediction: {prediction[0]}")
                    print(f"  Test probabilities: {probabilities[0]}")
                except Exception as e:
                    print(f"  Error making prediction: {e}")
            else:
                print(f"  Failed to load model")
                
                # Test fallback
                fallback_pred = get_fallback_prediction(test_features, disease)
                print(f"  Fallback prediction: {fallback_pred}")
        else:
            print(f"  Model file not found")

def test_predictions():
    """Test the prediction logic with sample data"""
    print("\n" + "="*50)
    print("TESTING PREDICTION LOGIC")
    print("="*50)
    
    # Sample patient data
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
    
    for patient in test_patients:
        print(f"\nTesting {patient['name']}:")
        features = patient['features']
        
        for disease in DISEASES:
            model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
            
            if os.path.exists(model_path):
                model_data = load_model_safely(model_path)
                if model_data:
                    model = model_data['model']
                    try:
                        prediction = model.predict([features])[0]
                        probabilities = model.predict_proba([features])[0]
                        confidence = max(probabilities)
                        
                        print(f"  {disease}: {prediction} (confidence: {confidence:.3f})")
                    except Exception as e:
                        print(f"  {disease}: ERROR - {e}")
                else:
                    fallback_pred = get_fallback_prediction(features, disease)
                    print(f"  {disease}: {fallback_pred} (fallback)")
            else:
                fallback_pred = get_fallback_prediction(features, disease)
                print(f"  {disease}: {fallback_pred} (fallback - no model)")

def test_api_simulation():
    """Simulate the API endpoint behavior"""
    print("\n" + "="*50)
    print("SIMULATING API ENDPOINT")
    print("="*50)
    
    # Import the endpoint function
    from app.api.v1.endpoints.predict import predict_all_diseases, PatientFeatures
    
    # Test data
    test_features = [45.0, 25.0, 3.5, 6.0, 4.0, 7.0, 8.0, 2.5, 75.0]
    patient = PatientFeatures(features=test_features)
    
    try:
        # This would normally be an async call, but we can test the logic
        import asyncio
        
        async def run_test():
            result = await predict_all_diseases(patient)
            return result
        
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_test())
        loop.close()
        
        print("API Response:")
        print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print(f"Error testing API simulation: {e}")

if __name__ == "__main__":
    try:
        print("ML Model Testing Suite")
        print("=" * 50)
        
        test_model_loading()
        test_predictions()
        test_api_simulation()
        
        print("\n" + "="*50)
        print("Testing completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()