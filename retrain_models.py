"""Train ML models using our shared module and primary dataset."""
import os
import pickle
import sys

# Add parent directory to path to import our shared module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ml_models import SimpleRandomForest, load_primary_dataset, generate_synthetic_features

def train_models():
    """Train ML models for each disease."""
    print("Training ML Models")
    print("=" * 50)
    
    # Load the primary dataset
    print("Loading primary dataset...")
    primary_data = load_primary_dataset()
    
    if not primary_data:
        print("ERROR: Could not load primary dataset!")
        return False
    
    print(f"Loaded {len(primary_data)} patient records")
    
    # Generate synthetic features for training
    print("Generating synthetic features...")
    features, labels = generate_synthetic_features(primary_data)
    
    print(f"Generated features shape: {len(features)} x {len(features[0]) if features else 0}")
    
    # Create models directory
    models_dir = os.path.join(parent_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models will be saved to: {models_dir}")
    
    # Disease names
    diseases = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']
    
    trained_models = {}
    
    for disease in diseases:
        print(f"\nTraining model for {disease}...")
        
        # Get labels for this disease
        disease_labels = labels[disease]
        
        # Create and train model
        model = SimpleRandomForest(n_trees=10, max_depth=5)
        model.fit(features, disease_labels)
        
        # Test the model on training data to get accuracy
        predictions = model.predict(features)
        correct = sum(1 for pred, true in zip(predictions, disease_labels) if pred == true)
        accuracy = correct / len(disease_labels)
        
        print(f"Training accuracy for {disease}: {accuracy:.3f}")
        
        # Save the model
        model_data = {
            'model': model,
            'accuracy': accuracy,
            'feature_names': ['age', 'bmi', 'symptom_duration', 'severity', 'progression', 'medication_response', 'exercise_tolerance', 'stress_impact', 'health_score']
        }
        
        model_path = os.path.join(models_dir, f'{disease}_ml_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        trained_models[disease] = {
            'model': model,
            'accuracy': accuracy,
            'path': model_path
        }
        
        print(f"Model saved to: {model_path}")
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"{'='*50}")
    
    print("\nModel Summary:")
    for disease, info in trained_models.items():
        print(f"{disease:12}: accuracy={info['accuracy']:.3f}")
    
    return True

if __name__ == "__main__":
    success = train_models()
    if success:
        print("\nAll models trained successfully!")
    else:
        print("\nTraining failed!")