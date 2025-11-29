"""Create simple rule-based model objects for each symptom and pickle them.

These are deterministic rule-based predictors (not ML-trained). They are used
as a fallback when scikit-learn training isn't available in the environment.
"""
import os
import pickle

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


class RuleModel:
    def __init__(self, name, rule_func, features):
        self.name = name
        self.rule_func = rule_func
        self.features = features

    def predict(self, X):
        # X can be list of dicts, list of lists, numpy array, or any 2D iterable
        preds = []
        for row in X:
            # allow dict-like rows
            if isinstance(row, dict):
                val = self.rule_func(row)
            else:
                # assume list/ndarray with same order as features
                d = {f: row[i] for i, f in enumerate(self.features)}
                val = self.rule_func(d)
            preds.append(1 if val else 0)
        return preds


def get_val(row, key):
    v = row.get(key, '')
    try:
        if isinstance(v, str):
            s = v.strip()
            up = s.upper()
            if up == 'TRUE':
                return 1.0
            if up == 'FALSE':
                return 0.0
            if s == '':
                return 0.0
        return float(v)
    except Exception:
        return 0.0


def diplopia_rule(r):
    return get_val(r, 'diplopia_severity_0_10') >= 1


def bulbar_rule(r):
    return get_val(r, 'bulbar_swallowing_severity') > 2


def facial_rule(r):
    return get_val(r, 'face_expression_limitation_score') >= 2


def fatigue_rule(r):
    return get_val(r, 'fatigue_severity_scale') >= 4


def limb_rule(r):
    return get_val(r, 'limb_activity_weakness_progression_0_10') >= 3


def ptosis_rule(r):
    return get_val(r, 'ptosis_degree_mm') >= 2


def respiratory_rule(r):
    return get_val(r, 'resp_fvc_percent') < 80


def make_rules(features):
    return {
        'diplopia': diplopia_rule,
        'bulbar': bulbar_rule,
        'facial': facial_rule,
        'fatigue': fatigue_rule,
        'limb': limb_rule,
        'ptosis': ptosis_rule,
        'respiratory': respiratory_rule,
    }


def main():
    # infer features by reading an existing primary_dataset.csv header if present
    data_path = os.path.join(ROOT, 'data', 'primary_dataset.csv')
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
        drop = set(['primary_id', 'visit_date'])
        features = [c for c in header if c not in drop]
    else:
        features = []

    rule_funcs = make_rules(features)
    for name, func in rule_funcs.items():
        model = RuleModel(name, func, features)
        path = os.path.join(MODELS_DIR, f"{name}_model.pkl")
        with open(path, 'wb') as f:
            pickle.dump({'model': model, 'features': features}, f)
        print(f"Wrote rule model pickle: {path}")


if __name__ == '__main__':
    main()
