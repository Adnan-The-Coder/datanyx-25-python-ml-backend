"""Simple test script to verify ML models work without FastAPI dependency."""
import os
import pickle
import sys

# Add the fast-api-backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Import our ML model classes
from ml_models import SimpleRandomForest, SimpleDecisionTree

MODELS_DIR = os.path.join(os.path.dirname(backend_dir), 'models')
DISEASES = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']

def test_ml_predictions():
    """Test ML model predictions with sample data."""
    print("Testing ML models...")
    
    # Sample patient features
    test_features = [45.0, 25.0, 7.5, 6.0, 5.0, 8.0, 3.0, 3.5, 60.0]
    print(f"Test features: {test_features}")
    print()
    
    predictions = {}
    
    for disease in DISEASES:
        model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    obj = pickle.load(f)
                
                model = obj['model']
                accuracy = obj.get('accuracy', 'unknown')
                
                # Make prediction
                pred = model.predict([test_features])[0]
                predictions[disease] = int(pred)
                
                print(f"{disease:12}: {pred} (accuracy: {accuracy:.3f})")
                
            except Exception as e:
                print(f"{disease:12}: ERROR - {e}")
                predictions[disease] = None
        else:
            print(f"{disease:12}: Model file not found")
            predictions[disease] = None
    
    print(f"\nSummary predictions: {predictions}")
    return predictions

def test_multiple_patients():
    """Test predictions for multiple patient scenarios."""
    print("\n" + "="*50)
    print("Testing multiple patient scenarios:")
    print("="*50)
    
    scenarios = [
        {"name": "Healthy patient", "features": [30.0, 22.0, 0.5, 0.5, 0.5, 2.0, 9.0, 0.5, 95.0]},
        {"name": "Mild symptoms", "features": [40.0, 24.0, 3.0, 2.5, 2.0, 5.0, 6.0, 1.5, 85.0]},
        {"name": "Severe symptoms", "features": [60.0, 28.0, 8.0, 8.0, 8.0, 9.0, 2.0, 4.0, 50.0]},
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"Features: {scenario['features']}")
        
        for disease in DISEASES:
            model_path = os.path.join(MODELS_DIR, f'{disease}_ml_model.pkl')
            
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        obj = pickle.load(f)
                    
                    model = obj['model']
                    pred = model.predict([scenario['features']])[0]
                    print(f"  {disease:12}: {'Present' if pred else 'Absent'}")
                    
                except Exception as e:
                    print(f"  {disease:12}: ERROR - {e}")

if __name__ == '__main__':
    print("ML Model Testing")
    print("=" * 50)
    
    # Test basic prediction
    test_ml_predictions()
    
    # Test multiple scenarios
    test_multiple_patients()
    
    print("\nTesting complete!")