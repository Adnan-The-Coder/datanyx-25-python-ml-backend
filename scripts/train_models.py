"""Train seven binary classifiers for the seven symptom groups and save pickles.

Rules used to derive binary labels (defaults; adjust as needed):
- diplopia: `diplopia_severity_0_10` >= 1
- bulbar: `bulbar_swallowing_severity` > 2
- facial: `face_expression_limitation_score` >= 2
- fatigue: `fatigue_severity_scale` >= 4
- limb: `limb_activity_weakness_progression_0_10` >= 3
- ptosis: `ptosis_degree_mm` >= 2
- respiratory: `resp_fvc_percent` < 80

This script reads `data/primary_dataset.csv`, engineers numeric features, trains RandomForest models,
and writes pickles to `models/`.
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'primary_dataset.csv')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

SYMTOMS = {
    'diplopia': {
        'rule_col': 'diplopia_severity_0_10',
        'rule': lambda v: pd.to_numeric(v, errors='coerce').fillna(0).astype(float) >= 1
    },
    'bulbar': {
        'rule_col': 'bulbar_swallowing_severity',
        'rule': lambda v: pd.to_numeric(v, errors='coerce').fillna(0).astype(float) > 2
    },
    'facial': {
        'rule_col': 'face_expression_limitation_score',
        'rule': lambda v: pd.to_numeric(v, errors='coerce').fillna(0).astype(float) >= 2
    },
    'fatigue': {
        'rule_col': 'fatigue_severity_scale',
        'rule': lambda v: pd.to_numeric(v, errors='coerce').fillna(0).astype(float) >= 4
    },
    'limb': {
        'rule_col': 'limb_activity_weakness_progression_0_10',
        'rule': lambda v: pd.to_numeric(v, errors='coerce').fillna(0).astype(float) >= 3
    },
    'ptosis': {
        'rule_col': 'ptosis_degree_mm',
        'rule': lambda v: pd.to_numeric(v, errors='coerce').fillna(0).astype(float) >= 2
    },
    'respiratory': {
        'rule_col': 'resp_fvc_percent',
        'rule': lambda v: pd.to_numeric(v, errors='coerce').fillna(999).astype(float) < 80
    }
}


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def create_labels(df):
    labels = {}
    for name, spec in SYMTOMS.items():
        col = spec['rule_col']
        if col not in df.columns:
            # create series of False if column missing
            labels[name] = pd.Series(False, index=df.index)
        else:
            labels[name] = spec['rule'](df[col])
    return labels


def preprocess_features(df):
    # drop identifiers and target-like columns
    drop_cols = ['primary_id', 'visit_date']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # remove any existing symptom indicator columns if present
    for s in SYMTOMS.keys():
        if s in X.columns:
            X = X.drop(columns=[s])
    # convert boolean-like columns 'TRUE'/'FALSE' and '0'/'1'
    def convert_series(s):
        if s.dtype == object:
            low = s.str.upper().map({'TRUE': 1, 'FALSE': 0})
            if low.notna().sum() > 0:
                return low
            # try numeric
            return pd.to_numeric(s, errors='coerce')
        return s

    X = X.apply(convert_series)
    # Fill numeric NaNs with median
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            med = X[col].median()
            X[col] = X[col].fillna(med)
        else:
            # for any remaining non-numeric, try factorize
            X[col], _ = pd.factorize(X[col].astype(str))
    return X.astype(float)


def train_and_save(df):
    labels = create_labels(df)
    X = preprocess_features(df)

    stats = {}
    for name, yser in labels.items():
        y = yser.astype(int)
        # simple train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        model_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({'model': clf, 'features': list(X.columns)}, f)
        stats[name] = {'accuracy': float(acc), 'path': model_path}
        print(f"Trained {name}: acc={acc:.3f} -> {model_path}")

    # Optionally, write labels back to primary csv
    for name, ser in labels.items():
        df[name] = ser.astype(int)
    out = DATA_PATH
    df.to_csv(out, index=False)
    print(f"Updated primary dataset with labels at: {out}")
    return stats


if __name__ == '__main__':
    print('Loading data...')
    df = load_data()
    print('Training models...')
    stats = train_and_save(df)
    print('Done. Models:')
    for k, v in stats.items():
        print(f" - {k}: {v}")
