"""Load pickled models, run predictions for each patient, and update primary_dataset.csv.

This script expects pickles in `models/` created by `train_models.py` where each pickle is a dict
with keys `model` and `features`.
"""
import os
import pickle
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'primary_dataset.csv')
MODELS_DIR = os.path.join(ROOT, 'models')

MODEL_FILES = [
    'diplopia_model.pkl',
    'bulbar_model.pkl',
    'facial_model.pkl',
    'fatigue_model.pkl',
    'limb_model.pkl',
    'ptosis_model.pkl',
    'respiratory_model.pkl',
]


def load_models():
    models = {}
    for fname in MODEL_FILES:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, 'rb') as f:
            models[fname.replace('_model.pkl', '')] = pickle.load(f)
    return models


def prepare_X(df, features):
    X = df.copy()
    # ensure feature presence
    for c in features:
        if c not in X.columns:
            X[c] = 0
    X = X[features]
    # coerce to numeric, fill na
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    return X


def run_all():
    df = pd.read_csv(DATA_PATH)
    models = load_models()
    for name, obj in models.items():
        model = obj['model']
        features = obj['features']
        X = prepare_X(df, features)
        preds = model.predict(X)
        df[name] = preds.astype(int)
        print(f"Wrote predictions for {name} (n={len(preds)})")
    df.to_csv(DATA_PATH, index=False)
    print(f"Updated primary dataset at {DATA_PATH}")


if __name__ == '__main__':
    run_all()
