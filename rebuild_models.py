import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

DISEASES = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']

FEATURE_NAMES = [
    'age', 'bmi', 'symptom_duration', 'severity', 
    'progression', 'medication_response', 'exercise_tolerance', 
    'stress_impact', 'health_score'
]

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    X = np.array([
        np.random.uniform(20, 80, n_samples),      # age
        np.random.uniform(18, 35, n_samples),      # bmi
        np.random.uniform(0.1, 10, n_samples),     # symptom_duration
        np.random.uniform(0.1, 10, n_samples),     # severity
        np.random.uniform(0.1, 10, n_samples),     # progression
        np.random.uniform(1, 10, n_samples),       # medication_response
        np.random.uniform(1, 10, n_samples),       # exercise_tolerance
        np.random.uniform(0.1, 5, n_samples),      # stress_impact
        np.random.uniform(50, 100, n_samples)      # health_score
    ]).T
    return X

def generate_disease_labels(X, disease):
    n_samples = X.shape[0]
    disease_weights = {
        'diplopia': [0.02, 0.01, 0.15, 0.20, 0.18, -0.10, -0.12, 0.15, -0.08],
        'bulbar': [0.03, 0.01, 0.18, 0.25, 0.20, -0.08, -0.15, 0.12, -0.10],
        'facial': [0.02, 0.005, 0.12, 0.22, 0.16, -0.09, -0.10, 0.10, -0.07],
        'fatigue': [0.01, 0.02, 0.10, 0.30, 0.15, -0.15, -0.20, 0.18, -0.12],
        'limb': [0.025, 0.01, 0.14, 0.28, 0.22, -0.12, -0.18, 0.14, -0.09],
        'ptosis': [0.03, 0.005, 0.16, 0.24, 0.19, -0.08, -0.10, 0.11, -0.08],
        'respiratory': [0.04, 0.02, 0.20, 0.35, 0.25, -0.20, -0.25, 0.20, -0.15]
    }
    weights = np.array(disease_weights.get(disease, [0.02] * 9))
    risk_scores = np.dot(X, weights)
    noise = np.random.normal(0, 0.1, n_samples)
    risk_scores += noise
    threshold = np.percentile(risk_scores, 70)
    y = (risk_scores > threshold).astype(int)
    return y

def train_model_for_disease(disease, X, y):
    print(f"Training model for {disease}...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{disease} model accuracy: {accuracy:.3f}")
    feature_importance = dict(zip(FEATURE_NAMES, model.feature_importances_))
    return model, accuracy, feature_importance

def save_model(disease, model, accuracy, feature_importance):
    model_data = {
        'model': model,
        'accuracy': accuracy,
        'disease': disease,
        'feature_names': FEATURE_NAMES,
        'feature_importance': feature_importance,
        'model_type': 'RandomForestClassifier',
        'sklearn_version': '1.0+'
    }
    model_path_pkl = os.path.join(models_dir, f'{disease}_ml_model.pkl')
    model_path_joblib = os.path.join(models_dir, f'{disease}_ml_model.joblib')
    try:
        with open(model_path_pkl, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {disease} model to {model_path_pkl}")
        joblib.dump(model_data, model_path_joblib)
        print(f"Saved {disease} model to {model_path_joblib}")
    except Exception as e:
        print(f"Error saving {disease} model: {e}")

def main():
    print("Rebuilding ML models for disease prediction...")
    print(f"Models will be saved to: {models_dir}")
    X = generate_synthetic_data(n_samples=2000)
    print(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    all_accuracies = {}
    
    for disease in DISEASES:
        try:
            y = generate_disease_labels(X, disease)
            positive_rate = np.mean(y)
            print(f"{disease}: {positive_rate:.1%} positive cases")
            model, accuracy, feature_importance = train_model_for_disease(disease, X, y)
            all_accuracies[disease] = accuracy
            save_model(disease, model, accuracy, feature_importance)
        except Exception as e:
            print(f"Error training model for {disease}: {e}")
            continue
    
    print("\nMODEL TRAINING SUMMARY")
    for disease, accuracy in all_accuracies.items():
        print(f"{disease}: {accuracy:.3f}")
    avg_accuracy = np.mean(list(all_accuracies.values()))
    print(f"Average: {avg_accuracy:.3f}")
    print(f"\nAll models saved to: {models_dir}")
    print("Models are now ready for deployment!")

def test_model_loading():
    print("\nTesting model loading...")
    test_features = [[45.0, 25.0, 3.5, 6.0, 4.0, 7.0, 8.0, 2.5, 75.0]]
    for disease in DISEASES:
        model_path = os.path.join(models_dir, f'{disease}_ml_model.pkl')
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                model = model_data['model']
                prediction = model.predict(test_features)
                probability = model.predict_proba(test_features)
                print(f"{disease}: prediction={prediction[0]}, probability={probability[0]}")
            except Exception as e:
                print(f"Error loading {disease} model: {e}")
        else:
            print(f"Model file not found for {disease}")

if __name__ == "__main__":
    try:
        import sklearn
        import joblib
        print(f"Using scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("Error: scikit-learn and joblib are required.")
        print("Install them with: pip install scikit-learn joblib")
        sys.exit(1)
    
    main()
    test_model_loading()