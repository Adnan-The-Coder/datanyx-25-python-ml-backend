"""Train models without pandas (uses csv + numpy + sklearn).
This is a fallback when pandas isn't importable.
"""
import os
import csv
import pickle
import numpy as np
from collections import defaultdict
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
        'rule': lambda v: np.nan_to_num(v).astype(float) >= 1
    },
    'bulbar': {
        'rule_col': 'bulbar_swallowing_severity',
        'rule': lambda v: np.nan_to_num(v).astype(float) > 2
    },
    'facial': {
        'rule_col': 'face_expression_limitation_score',
        'rule': lambda v: np.nan_to_num(v).astype(float) >= 2
    },
    'fatigue': {
        'rule_col': 'fatigue_severity_scale',
        'rule': lambda v: np.nan_to_num(v).astype(float) >= 4
    },
    'limb': {
        'rule_col': 'limb_activity_weakness_progression_0_10',
        'rule': lambda v: np.nan_to_num(v).astype(float) >= 3
    },
    'ptosis': {
        'rule_col': 'ptosis_degree_mm',
        'rule': lambda v: np.nan_to_num(v).astype(float) >= 2
    },
    'respiratory': {
        'rule_col': 'resp_fvc_percent',
        'rule': lambda v: np.nan_to_num(v).astype(float) < 80
    }
}


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        rows = [r for r in rdr]
        fieldnames = rdr.fieldnames
    return fieldnames, rows


def to_numeric_column(rows, col, cat_maps):
    vals = []
    for r in rows:
        v = r.get(col, '')
        if isinstance(v, str):
            s = v.strip()
            if s == '':
                vals.append(np.nan)
                continue
            up = s.upper()
            if up == 'TRUE':
                vals.append(1.0)
                continue
            if up == 'FALSE':
                vals.append(0.0)
                continue
            try:
                fv = float(s)
                vals.append(fv)
                continue
            except Exception:
                # factorize
                m = cat_maps[col]
                if s not in m:
                    m[s] = len(m) + 1
                vals.append(float(m[s]))
                continue
        else:
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)
    return np.array(vals, dtype=float)


def build_feature_matrix(fieldnames, rows):
    drop_cols = set(['primary_id', 'visit_date'])
    features = [c for c in fieldnames if c not in drop_cols]
    # do not include label columns (symptom names) if present
    for s in SYMTOMS.keys():
        if s in features:
            features.remove(s)

    cat_maps = defaultdict(dict)
    cols_data = {}
    for c in features:
        cols_data[c] = to_numeric_column(rows, c, cat_maps)

    # fill na with median
    for c, arr in cols_data.items():
        mask = np.isnan(arr)
        if mask.any():
            med = np.nanmedian(arr)
            if np.isnan(med):
                med = 0.0
            arr[mask] = med
            cols_data[c] = arr

    X = np.vstack([cols_data[c] for c in features]).T if features else np.zeros((len(rows), 0))
    return features, X


def create_labels(rows):
    labels = {}
    # For rule evaluations, create numeric arrays for rule_col
    for name, spec in SYMTOMS.items():
        col = spec['rule_col']
        arr = []
        for r in rows:
            v = r.get(col, '')
            if isinstance(v, str):
                s = v.strip()
                if s == '':
                    arr.append(np.nan)
                    continue
                up = s.upper()
                if up == 'TRUE':
                    arr.append(1.0)
                    continue
                if up == 'FALSE':
                    arr.append(0.0)
                    continue
                try:
                    arr.append(float(s))
                except Exception:
                    arr.append(np.nan)
            else:
                try:
                    arr.append(float(v))
                except Exception:
                    arr.append(np.nan)
        arr = np.array(arr, dtype=float)
        labels[name] = spec['rule'](arr)
    return labels


def train_and_save():
    fieldnames, rows = read_csv(DATA_PATH)
    features, X = build_feature_matrix(fieldnames, rows)
    labels = create_labels(rows)
    stats = {}
    for name, ybool in labels.items():
        y = ybool.astype(int)
        if X.shape[1] == 0:
            # no features, use constant model
            from sklearn.dummy import DummyClassifier
            clf = DummyClassifier(strategy='most_frequent')
            clf.fit(np.zeros((len(y), 1)), y)
            acc = clf.score(np.zeros((len(y), 1)), y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            acc = accuracy_score(y_test, preds)
        model_path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({'model': clf, 'features': features}, f)
        stats[name] = {'accuracy': float(acc), 'path': model_path}
        print(f"Trained {name}: acc={acc:.3f} -> {model_path}")

    # write labels back into CSV
    out_rows = []
    for i, r in enumerate(rows):
        newr = dict(r)
        for name in labels.keys():
            newr[name] = int(labels[name][i])
        out_rows.append(newr)

    # ensure header includes new label columns
    out_fieldnames = list(fieldnames)
    for name in labels.keys():
        if name not in out_fieldnames:
            out_fieldnames.append(name)

    with open(DATA_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"Updated primary dataset with labels at: {DATA_PATH}")
    return stats


if __name__ == '__main__':
    stats = train_and_save()
    print('Done')
