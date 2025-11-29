"""Run model pickles to predict 7 symptoms and produce a compact primary dataset.

Behavior:
- Reads `data/primary_dataset.csv` (merged data with many columns).
- For each symptom tries to load `models/{symptom}_model.pkl` and call `predict`.
- If unpickling or prediction fails, falls back to deterministic rule computed from CSV values.
- Writes `data/primary_compact.csv` with columns: `primary_id`, and the 7 symptoms (0/1).
"""
import os
import csv
import pickle
import traceback

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'primary_dataset.csv')
OUT_PATH = os.path.join(ROOT, 'data', 'primary_compact.csv')
MODELS_DIR = os.path.join(ROOT, 'models')

SYMPTOMS = ['diplopia', 'bulbar', 'facial', 'fatigue', 'limb', 'ptosis', 'respiratory']


def get_val(row, key):
    v = row.get(key, '')
    if v is None:
        return 0.0
    s = str(v).strip()
    if s == '':
        return 0.0
    up = s.upper()
    if up == 'TRUE':
        return 1.0
    if up == 'FALSE':
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def rule_for(symptom, row):
    if symptom == 'diplopia':
        return int(get_val(row, 'diplopia_severity_0_10') >= 1)
    if symptom == 'bulbar':
        return int(get_val(row, 'bulbar_swallowing_severity') > 2)
    if symptom == 'facial':
        return int(get_val(row, 'face_expression_limitation_score') >= 2)
    if symptom == 'fatigue':
        return int(get_val(row, 'fatigue_severity_scale') >= 4)
    if symptom == 'limb':
        return int(get_val(row, 'limb_activity_weakness_progression_0_10') >= 3)
    if symptom == 'ptosis':
        return int(get_val(row, 'ptosis_degree_mm') >= 2)
    if symptom == 'respiratory':
        return int(get_val(row, 'resp_fvc_percent') < 80)
    return 0


def try_load_model(symptom):
    path = os.path.join(MODELS_DIR, f"{symptom}_model.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        # obj may be dict with 'model' and 'features' or a sklearn estimator directly
        if isinstance(obj, dict) and 'model' in obj and 'features' in obj:
            return obj
        # try well-known structure
        if hasattr(obj, 'predict'):
            return {'model': obj, 'features': None}
        return None
    except Exception:
        # unpickling failed
        return None


def prepare_feature_matrix(rows, features):
    # build list of lists for each row in order
    X = []
    for r in rows:
        rowvec = []
        for f in features:
            rowvec.append(get_val(r, f))
        X.append(rowvec)
    return X


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        rows = [r for r in rdr]

    results = {s: [] for s in SYMPTOMS}
    methods = {}

    for symptom in SYMPTOMS:
        obj = try_load_model(symptom)
        if obj is None:
            methods[symptom] = 'rule'
            # apply rule per row
            for r in rows:
                results[symptom].append(rule_for(symptom, r))
            continue

        # if features specified, prepare matrix
        feats = obj.get('features')
        model = obj.get('model')
        if feats:
            X = prepare_feature_matrix(rows, feats)
        else:
            # if features unknown, try to call model.predict with full row dicts
            X = rows

        try:
            preds = model.predict(X)
            # ensure ints
            preds = [int(p) for p in preds]
            results[symptom] = preds
            methods[symptom] = 'pickle'
        except Exception:
            # fallback to rule
            methods[symptom] = 'rule-fallback'
            for r in rows:
                results[symptom].append(rule_for(symptom, r))

    # write compact CSV: primary_id + symptom columns
    out_fields = ['primary_id'] + SYMPTOMS
    with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for i, r in enumerate(rows):
            out = {'primary_id': r.get('primary_id') or r.get('patient_id')}
            for s in SYMPTOMS:
                out[s] = results[s][i]
            w.writerow(out)

    print('Wrote compact primary dataset:', OUT_PATH)
    print('Methods used per symptom:')
    for s in SYMPTOMS:
        print(f' - {s}: {methods.get(s)}')


if __name__ == '__main__':
    main()
