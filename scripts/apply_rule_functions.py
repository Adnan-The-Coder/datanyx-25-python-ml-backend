"""Apply deterministic rules directly to `primary_dataset.csv` to produce symptom columns.

This avoids unpickling issues by computing predictions from values in the CSV.
"""
import os
import csv

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'primary_dataset.csv')


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


def diplopia_rule(row):
    return get_val(row, 'diplopia_severity_0_10') >= 1


def bulbar_rule(row):
    return get_val(row, 'bulbar_swallowing_severity') > 2


def facial_rule(row):
    return get_val(row, 'face_expression_limitation_score') >= 2


def fatigue_rule(row):
    return get_val(row, 'fatigue_severity_scale') >= 4


def limb_rule(row):
    return get_val(row, 'limb_activity_weakness_progression_0_10') >= 3


def ptosis_rule(row):
    return get_val(row, 'ptosis_degree_mm') >= 2


def respiratory_rule(row):
    return get_val(row, 'resp_fvc_percent') < 80


RULES = {
    'diplopia': diplopia_rule,
    'bulbar': bulbar_rule,
    'facial': facial_rule,
    'fatigue': fatigue_rule,
    'limb': limb_rule,
    'ptosis': ptosis_rule,
    'respiratory': respiratory_rule,
}


def apply_rules():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        rdr = csv.DictReader(f)
        rows = [r for r in rdr]
        fieldnames = list(rdr.fieldnames)

    for name, fn in RULES.items():
        if name not in fieldnames:
            fieldnames.append(name)
        for r in rows:
            r[name] = '1' if fn(r) else '0'

    with open(DATA_PATH, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Applied rules and updated {DATA_PATH}")


if __name__ == '__main__':
    apply_rules()
