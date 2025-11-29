"""Quick API test without needing a running server."""
import os
import pickle
import sys
import json

# Add parent directory for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from ml_models import SimpleRandomForest

def test_models_directly():
    """Test the ML models directly without HTTP API."""
    print("Direct ML Model Testing")
    print("=" * 50)
    
    models_dir = os.path.join(script_dir, '..', 'models')
    models_dir = os.path.normpath(models_dir)
    diseases = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']
    
    # Test patient features
    test_features = [45.0, 25.0, 7.5, 6.0, 5.0, 8.0, 3.0, 3.5, 60.0]
    
    print("Testing Models:")
    print(f"Patient features: {test_features}")
    print(f"Models directory: {models_dir}")
    print()
    
    results = {}
    
    for disease in diseases:
        model_path = os.path.join(models_dir, f'{disease}_ml_model.pkl')
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                model = model_data['model']
                accuracy = model_data.get('accuracy', 'unknown')
                
                prediction = model.predict([test_features])[0]
                status = "Present" if prediction == 1 else "Absent"
                
                results[disease] = {
                    "prediction": int(prediction),
                    "status": status,
                    "accuracy": f"{accuracy:.3f}" if isinstance(accuracy, float) else str(accuracy)
                }
                
                print(f"{disease:12}: {status:7} (prediction: {prediction}, accuracy: {results[disease]['accuracy']})")
                
            except Exception as e:
                results[disease] = {"error": str(e)}
                print(f"{disease:12}: ERROR - {e}")
        else:
            results[disease] = {"error": "Model file not found"}
            print(f"{disease:12}: ERROR - Model file not found")
    
    print("\n" + "=" * 50)
    print("Direct Testing Complete!")
    
    # Simulate API response format
    api_response = {
        "patient_features": test_features,
        "predictions": results,
        "api_simulation": True
    }
    
    print("\nAPI Response Format:")
    print(json.dumps(api_response, indent=2))
    
    return results

if __name__ == "__main__":
    test_models_directly()