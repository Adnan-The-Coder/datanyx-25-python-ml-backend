"""Merge the seven symptom CSVs into `data/primary_dataset.csv` without pandas.
Uses `patient_id` as primary key.
"""
import os
import csv

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
FILES = [
    'Diplopia _ Double Vision 2 - Sheet1.csv',
    'bulbar_symptoms_dataset.csv',
    'facial_neck_weakness_dataset.csv',
    'fatigue_endurance_dataset.csv',
    'limb_weakness_dataset.csv',
    'Ptosis _ Eyelid Drooping - Sheet1.csv',
    'respiratory_difficulty_dataset.csv',
]


def merge():
    records = {}
    all_cols = []
    cols_set = set()
    for fname in FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print('Missing', path)
            continue
        with open(path, newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                pid = row.get('patient_id')
                if not pid:
                    continue
                rec = records.setdefault(pid, {})
                if 'visit_date' not in rec and row.get('visit_date'):
                    rec['visit_date'] = row.get('visit_date')
                for k, v in row.items():
                    if k in ('patient_id', 'visit_date'):
                        continue
                    colname = k.strip()
                    # handle collisions by prefixing with filename base if needed
                    if colname in rec and rec[colname] != v:
                        base = os.path.splitext(fname)[0].replace(' ', '_').replace('-', '_')
                        colname = f"{base}_{colname}"
                    if colname not in cols_set:
                        cols_set.add(colname)
                        all_cols.append(colname)
                    rec[colname] = v

    out_path = os.path.join(DATA_DIR, 'primary_dataset.csv')
    header = ['primary_id', 'visit_date'] + all_cols
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for pid in sorted(records.keys()):
            row = {'primary_id': pid}
            row['visit_date'] = records[pid].get('visit_date', '')
            for c in all_cols:
                row[c] = records[pid].get(c, '')
            w.writerow(row)

    print('Wrote', out_path)


if __name__ == '__main__':
    merge()
