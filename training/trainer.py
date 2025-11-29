import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # Simple replacement for LightGBM/LSTM for quick template build

# Ensure model directory exists
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'app', 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'mg_clinical_data.csv')
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. SIMULATED DATA GENERATION ---
def generate_dummy_data():
    """Generates a small synthetic dataset for training demonstration."""
    N = 100
    data = {
        'Age': np.random.randint(40, 75, N),
        'Ptosis_severity_score': np.random.randint(1, 4, N),
        'Grip_strength_kg': np.random.uniform(5, 40, N),
        'Respiratory_rate': np.random.uniform(12, 30, N),
        # Target labels (Simulated)
        'Diagnosis_Label': np.random.choice([0, 1, 2, 3, 4], N, p=[0.4, 0.2, 0.15, 0.1, 0.15]), # 0=MG, 1=LEMS, etc.
        'Progression_Label': np.random.choice([0, 1, 2], N, p=[0.6, 0.3, 0.1]) # 0=Fluctuating, 1=Ascending, 2=Decline
    }
    df = pd.DataFrame(data)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dummy dataset generated and saved to {DATA_PATH}")
    return df

# --- 2. PRE-PROCESSING / FEATURE ENGINEERING ---
def preprocess_and_engineer_features(df):
    """Placeholder for your robust cleaning and feature engineering (SSI, FBS)."""
    
    # Selecting simple features for model training demonstration
    X = df[['Age', 'Ptosis_severity_score', 'Grip_strength_kg', 'Respiratory_rate']]
    
    # Standardization (Scaler Training)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the fitted scaler (Critical for FeatureService to use)
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'feature_scaler.pkl'))
    print(f"Scaler saved to {MODEL_DIR}/feature_scaler.pkl")

    return X_scaled, df['Diagnosis_Label'], df['Progression_Label']

# --- 3. MODEL TRAINING FUNCTIONS ---
def train_diagnosis_model(X, y_diag):
    """Trains the Random Forest Diagnosis Classifier."""
    print("Training Diagnosis Model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_diag)
    joblib.dump(model, os.path.join(MODEL_DIR, 'diagnosis_rf.pkl'))
    print(f"Diagnosis Model saved to {MODEL_DIR}/diagnosis_rf.pkl")
    return "Diagnosis Model (Random Forest) trained successfully."

def train_severity_model(X, y_prog):
    """Trains the Severity Prediction Model (using LogisticRegression placeholder)."""
    print("Training Severity Model (Logistic Regression Placeholder)...")
    # In a real scenario, this would be LightGBM or CatBoost
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X, y_prog)
    joblib.dump(model, os.path.join(MODEL_DIR, 'severity_lgbm.pkl'))
    print(f"Severity Model saved to {MODEL_DIR}/severity_lgbm.pkl")
    return "Severity Model (LightGBM Placeholder) trained successfully."

# --- MAIN TRAINING PIPELINE ---
def run_training_pipeline(model_name: str = 'all'):
    """
    Runs the full training pipeline or a specific module.
    
    :param model_name: 'diagnosis', 'severity', or 'all'.
    """
    try:
        # Load or generate data
        if not os.path.exists(DATA_PATH):
            df = generate_dummy_data()
        else:
            df = pd.read_csv(DATA_PATH)

        X, y_diag, y_prog = preprocess_and_engineer_features(df)
        
        results = []

        if model_name in ['all', 'diagnosis']:
            results.append(train_diagnosis_model(X, y_diag))

        if model_name in ['all', 'severity']:
            results.append(train_severity_model(X, y_prog))

        # Placeholder for LSTM/Cox model (requires time-series data and Keras)
        if model_name in ['all', 'warning']:
             results.append("Warning Model (LSTM/Cox) training skipped in template.")

        return {"status": "success", "trained_models": results}

    except Exception as e:
        return {"status": "error", "message": f"Training failed: {e}"}

if __name__ == '__main__':
    # This block allows local execution of the trainer script
    result = run_training_pipeline()
    print(result)