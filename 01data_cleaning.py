"""
Diabetes 130-US Hospitals - Data Cleaning Pipeline
Dataset: UCI ML Repository (Strack et al., 2014)
Goal: Prepare clean data for readmission analysis & modeling
"""
import pandas as pd
import numpy as np

# ── Load raw data ────────────────────────────────────────────────
df = pd.read_csv('diabetic_data.csv')
id_map = pd.read_csv('IDS_mapping.csv')
print(f"Raw shape: {df.shape}")

# Replace '?' placeholder with proper NaN
df = df.replace('?', np.nan)

# ── Decision 1: Drop columns that are almost entirely missing ───
# 'weight' is 97% missing — not usable as a feature
df = df.drop(columns=['weight'])

# ── Decision 2: Recode high-missing lab columns as meaningful ───
# max_glu_serum / A1Cresult being missing usually means "test not performed"
# rather than random missingness — this itself is a meaningful clinical signal
df['max_glu_serum'] = df['max_glu_serum'].fillna('Not Tested')
df['A1Cresult'] = df['A1Cresult'].fillna('Not Tested')

# ── Decision 3: Moderate-missing categoricals -> explicit "Unknown" ──
df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
df['payer_code'] = df['payer_code'].fillna('Unknown')
df['race'] = df['race'].fillna('Unknown')

# ── Decision 4: Diagnosis codes ───────────────────────────────────
# diag_1 (primary) missing is negligible (~0.02%) -> safe to drop those rows
df = df.dropna(subset=['diag_1'])

# diag_2 / diag_3 (secondary/tertiary) missing means "no second/third
# diagnosis was logged" -> informative, not random, so keep as "Unknown"
df['diag_2'] = df['diag_2'].fillna('Unknown')
df['diag_3'] = df['diag_3'].fillna('Unknown')

# Group raw ICD-9 codes into broad clinical categories.
# Hundreds of raw codes (250.83, 276, V57...) are unusable as model features
# on their own, so we bucket them the way Strack et al. (2014) did.
def icd9_to_category(code):
    if code == 'Unknown':
        return 'Unknown'
    if str(code).startswith('V'):
        return 'Supplemental (V-code)'
    if str(code).startswith('E'):
        return 'External cause (E-code)'
    try:
        val = float(code)
    except ValueError:
        return 'Other'
    if val == 250 or (250 <= val < 251):
        return 'Diabetes'
    if (390 <= val <= 459) or val == 785:
        return 'Circulatory'
    if (460 <= val <= 519) or val == 786:
        return 'Respiratory'
    if (520 <= val <= 579) or val == 787:
        return 'Digestive'
    if (800 <= val <= 999):
        return 'Injury/Poisoning'
    if (710 <= val <= 739):
        return 'Musculoskeletal'
    if (580 <= val <= 629) or val == 788:
        return 'Genitourinary'
    if (140 <= val <= 239):
        return 'Neoplasms'
    return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[f'{col}_category'] = df[col].apply(icd9_to_category)

# ── Decision 5: Remove expired/hospice patients ──────────────────
# discharge_disposition_id 11,13,14,19,20,21 = deceased or hospice
# These patients cannot be "readmitted", so including them biases the target
expired_codes = [11, 13, 14, 19, 20, 21]
before = len(df)
df = df[~df['discharge_disposition_id'].isin(expired_codes)]
print(f"Removed {before - len(df)} expired/hospice encounters")

# ── Decision 6: Keep only first encounter per patient ────────────
# Prevents data leakage — same patient appearing in both train/test
# would let the model "memorize" rather than generalize
before = len(df)
df = df.sort_values('encounter_id').drop_duplicates(subset='patient_nbr', keep='first')
print(f"Removed {before - len(df)} repeat encounters (kept first per patient)")

# ── Decision 7: Merge in human-readable ID mappings ──────────────
def merge_mapping(df, id_col):
    sub = id_map[id_map.iloc[:, 0].astype(str).str.contains(id_col, na=False) == False]
    return df  # handled separately below since IDS_mapping.csv stacks 3 tables

# IDS_mapping.csv contains 3 stacked lookup tables (admission_type_id,
# discharge_disposition_id, admission_source_id) separated by blank rows.
# Split them out:
raw = pd.read_csv('IDS_mapping.csv', header=None, names=['id', 'description'])
names = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']
header_rows = raw[raw['id'].isin(names)].index.tolist()
split_points = header_rows + [len(raw)]

tables = {}
for i, name in enumerate(names):
    start = split_points[i] + 1
    end = split_points[i+1]
    t = raw.iloc[start:end].dropna()
    t['id'] = pd.to_numeric(t['id'], errors='coerce')
    t = t.dropna(subset=['id'])
    t['id'] = t['id'].astype(int)
    tables[name] = dict(zip(t['id'], t['description']))

df['admission_type'] = df['admission_type_id'].map(tables['admission_type_id'])
df['discharge_disposition'] = df['discharge_disposition_id'].map(tables['discharge_disposition_id'])
df['admission_source'] = df['admission_source_id'].map(tables['admission_source_id'])

# ── Decision 8: Create binary target variable ────────────────────
# Business question = "will this patient be readmitted within 30 days?"
df['readmitted_30d'] = (df['readmitted'] == '<30').astype(int)

# ── Save cleaned dataset ──────────────────────────────────────────
out_path = 'diabetic_data_cleaned.csv'
df.to_csv(out_path, index=False)

print(f"\nFinal shape: {df.shape}")
print(f"\nTarget distribution (readmitted_30d):")
print(df['readmitted_30d'].value_counts(normalize=True).round(3))
print(f"\nSaved cleaned data to: {out_path}")