"""Apply rule-based pickles to dataset and update primary_dataset.csv without pandas.
"""
import os
import csv
import pickle

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'primary_dataset.csv')
MODELS_DIR = os.path.join(ROOT, 'models')


def load_models():
    models = {}
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith('_model.pkl'):
            continue
        name = fname.replace('_model.pkl', '')
        with open(os.path.join(MODELS_DIR, fname), 'rb') as f:
            models[name] = pickle.load(f)
    return models


def apply_models():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        rows = [r for r in rdr]
        fieldnames = rdr.fieldnames

    models = load_models()
    for name, obj in models.items():
        model = obj['model']
        preds = model.predict(rows)
        for i, p in enumerate(preds):
            rows[i][name] = str(int(p))
        if name not in fieldnames:
            fieldnames.append(name)

    with open(DATA_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Updated {DATA_PATH} with rule predictions.")


if __name__ == '__main__':
    apply_models()
