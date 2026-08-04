"""
STEP 1 — DATA CLEANING (v3 — Production-Ready)
===============================================
Run this first:  python step1_clean_data_v3.py

What changed from v2:
  ✅ Null guard on estimated_labour_days before date fill
  ✅ Explicit median imputation for any remaining numeric nulls
  ✅ project_quarter kept as int (ordinal, not OHE — same logic as materials_grade)
  ✅ city_tier validation — prints unexpected cities so nothing silently misclassifies
  ✅ Dtype assertion before save — catches stray object columns that would break model
  ✅ Skewness check on target — reminds you to log-transform in step 2
  ✅ All v2 leakage fixes retained
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── CONFIG ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = DATA_DIR / "interior_india_10000_extended_enriched.csv"
CLEAN_FILE = DATA_DIR / "cleaned_data.csv"

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 65)
print("STEP 1: DATA CLEANING  ")
print("=" * 65)

# ── LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_FILE)
print(f"\n Loaded raw data: {df.shape[0]} rows × {df.shape[1]} columns")

# ── STEP 1: DROP FULLY EMPTY COLUMNS ──────────────────────────────────────
all_null_cols = [c for c in df.columns if df[c].isnull().all()]
df.drop(columns=all_null_cols, inplace=True)
print(f"\n Dropped {len(all_null_cols)} fully-empty columns (all NaN, zero info):")
for c in all_null_cols:
    print(f"     - {c}")

# ── STEP 2: NULL GUARD — estimated_labour_days ─────────────────────────────
# We rely on this column to fill missing end_dates. If it has nulls itself,
# the resulting project_duration_days will be silently NaN — catch it early.
if "estimated_labour_days" in df.columns:
    labour_nulls = df["estimated_labour_days"].isnull().sum()
    if labour_nulls > 0:
        print(f"\n estimated_labour_days has {labour_nulls} nulls — filling with median before use.")
        df["estimated_labour_days"].fillna(df["estimated_labour_days"].median(), inplace=True)
    else:
        print(f"\n estimated_labour_days: no nulls — safe to use for end_date fill.")

# ── STEP 3: RECOVER project_duration_days & project_quarter ───────────────
df["start_date"] = pd.to_datetime(df["start_date"])
df["end_date"]   = pd.to_datetime(df["end_date"])

# Fill missing end_date using start + estimated_labour_days
mask_missing = df["end_date"].isnull()
df.loc[mask_missing, "end_date"] = (
    df.loc[mask_missing, "start_date"] +
    pd.to_timedelta(df.loc[mask_missing, "estimated_labour_days"], unit="D")
)

# Rebuild duration from clean dates
df["project_duration_days"] = (df["end_date"] - df["start_date"]).dt.days.astype(int)

# Keep project_quarter as INT (1–4) — it is ordinal, same logic as materials_grade.
# OHE would lose the seasonal ordering (Q1 < Q2 < Q3 < Q4).
df["project_quarter"] = df["start_date"].dt.quarter  # int 1/2/3/4

print(f"\n🔧 Recovered project_duration_days (filled {mask_missing.sum()} missing end_dates)")
print(f"🔧 Kept project_quarter as ordinal int (1–4) — NOT one-hot encoded")

# ── STEP 4: DROP LEAKAGE COLUMNS ──────────────────────────────────────────
# These are all derived from total_cost_inr — including them is cheating.
leakage_cols = ["cost_per_sqft_inr", "cost_per_room_inr", "cost_category"]
dropped_leakage = [c for c in leakage_cols if c in df.columns]
df.drop(columns=dropped_leakage, inplace=True)
print(f"\n  Dropped {len(dropped_leakage)} leakage columns (derived from target):")
for c in dropped_leakage:
    print(f"     - {c}  ← derived from total_cost_inr")

# ── STEP 5: DROP IRRELEVANT / ADMIN / ZERO-VARIANCE COLUMNS ───────────────
irrelevant = [
    "project_id",         # unique identifier — no predictive value
    "latitude",           # city_tier captures geography better
    "longitude",          # same
    "image_url",          # all null (safety net)
    "notes",              # all null
    "provenance_sources", # single constant value — zero variance
    "start_date",         # now captured by project_quarter
    "end_date",           # duration already computed
    "room_types",         # redundant — rooms count already present
    "postal_code",        # all null
]
drop_irr = [c for c in irrelevant if c in df.columns]
df.drop(columns=drop_irr, inplace=True)
print(f"\n🗑  Dropped {len(drop_irr)} irrelevant/zero-variance/admin columns:")
for c in drop_irr:
    print(f"     - {c}")

# ── STEP 6: FEATURE ENGINEERING ───────────────────────────────────────────

# 6a. n_materials — count of material types used
df["n_materials"] = df["main_materials"].apply(
    lambda x: len(str(x).split(";")) if pd.notna(x) else 1
)
print(f"\n🔧 Engineered: n_materials (count of ';'-separated materials)")

# 6b. city_tier — based on pricing tier observed in Indian real-estate
metro = {"Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata"}
tier1 = {"Pune", "Ahmedabad", "Surat"}

# Validation: flag any city not in known sets (catches typos, new cities)
if "city" in df.columns:
    known_cities = metro | tier1
    # tier2 is a catch-all; we only warn if a city looks suspicious
    city_counts = df["city"].value_counts()
    unexpected = [c for c in city_counts.index if c not in known_cities]
    if unexpected:
        print(f"\n Cities classified as tier2 ({len(unexpected)} unique):")
        for c in unexpected[:10]:   # show first 10
            print(f"     - {c}  ({city_counts[c]} rows)")
        if len(unexpected) > 10:
            print(f"     ... and {len(unexpected) - 10} more")

df["city_tier"] = df["city"].apply(
    lambda c: "metro" if c in metro else ("tier1" if c in tier1 else "tier2")
)
print(f"\n Engineered: city_tier (metro / tier1 / tier2)")
print(f"\n   City tier distribution:")
print(df["city_tier"].value_counts().to_string())

# 6c. furniture_included → int
df["furniture_included"] = df["furniture_included"].astype(int)
print(f"\n Converted furniture_included: bool → int (0/1)")

# ── STEP 7: DROP RAW COLUMNS REPLACED BY ENGINEERED FEATURES ──────────────
df.drop(columns=["main_materials", "city", "state"], inplace=True)
print(f"\n  Dropped: main_materials, city, state (replaced by engineered features)")

# ── STEP 8: ORDINAL ENCODE materials_grade ────────────────────────────────
# Preserves natural cost hierarchy: economy < standard < premium < luxury
grade_map = {"economy": 1, "standard": 2, "premium": 3, "luxury": 4}
df["materials_grade"] = df["materials_grade"].map(grade_map)
print(f"\n Ordinal-encoded materials_grade: economy=1 / standard=2 / premium=3 / luxury=4")

# ── STEP 9: ONE-HOT ENCODE NOMINAL CATEGORICALS ───────────────────────────
# project_quarter intentionally excluded — it is ordinal (kept as int above)
ohe_cols = ["project_type", "scope", "contractor_type", "design_style", "city_tier"]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)
print(f"\n One-hot encoded (nominal only): {ohe_cols}")

# quoted_from → label-encode (7 categories, data-quality proxy signal)
if "quoted_from" in df.columns:
    df["quoted_from"] = df["quoted_from"].astype("category").cat.codes
    print(f" Label-encoded: quoted_from (7 sources → 0–6)")

# ── STEP 10: IMPUTE REMAINING MISSING VALUES ──────────────────────────────
# After all transforms, fill any remaining numeric nulls with column median.
# This prevents silent NaN propagation into the model.
num_cols = df.select_dtypes(include="number").columns.tolist()
null_counts = df[num_cols].isnull().sum()
cols_with_nulls = null_counts[null_counts > 0]

if not cols_with_nulls.empty:
    print(f"\n Imputing {len(cols_with_nulls)} column(s) with median:")
    for col, cnt in cols_with_nulls.items():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"     - {col}: {cnt} nulls → filled with median ({median_val:.2f})")
else:
    print(f"\n No missing values found in numeric columns — no imputation needed.")

# ── STEP 11: DTYPE VALIDATION ─────────────────────────────────────────────
# Any object column at this point means an encoding step was missed.
# Better to catch it here than get a cryptic model error later.
obj_cols = df.select_dtypes(include="object").columns.tolist()
if obj_cols:
    print(f"\n WARNING — object columns still present (will break model):")
    for c in obj_cols:
        print(f"     - {c}  | sample values: {df[c].unique()[:5].tolist()}")
    raise ValueError(
        f"Fix encoding for these columns before proceeding: {obj_cols}"
    )
else:
    print(f"\n Dtype check passed — all columns are numeric.")

# ── STEP 12: VERIFY ZERO MISSING VALUES ───────────────────────────────────
missing = df.isnull().sum().sum()
print(f"\n Missing values remaining: {missing}")
assert missing == 0, f"Still {missing} missing values — check steps above!"

# ── STEP 13: TARGET SKEWNESS CHECK ────────────────────────────────────────
target = "total_cost_inr"
skew = df[target].skew()
print(f"\n Target skewness: {skew:.2f}")
if skew > 1.0:
    print(f"     Right-skewed (skew > 1). Log-transform in step 2:")
    print(f"      y = np.log1p(df['{target}'])")
    print(f"      Remember to inverse-transform predictions: np.expm1(y_pred)")
else:
    print(f"   Distribution looks acceptable (skew ≤ 1).")

# ── STEP 14: REORDER — TARGET AT END ──────────────────────────────────────
other_cols = [c for c in df.columns if c != target]
df = df[other_cols + [target]]

# ── SUMMARY ───────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"CLEANING COMPLETE")
print(f"{'=' * 65}")
print(f"Final shape    : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"All numeric    : {df.select_dtypes(include='object').empty}")
print(f"\nFinal feature columns:")
for i, c in enumerate(df.columns):
    tag = " ← TARGET" if c == target else ""
    print(f"  {i+1:2d}. {c}{tag}")

print(f"\nTarget variable (total_cost_inr) stats:")
print(df[target].describe().apply(lambda x: f"₹{x:,.0f}").to_string())

# ── SAVE ──────────────────────────────────────────────────────────────────
import time
for attempt in range(3):
    try:
        df.to_csv(CLEAN_FILE, index=False)
        print(f"\n✅ Saved cleaned data → {CLEAN_FILE}")
        break
    except PermissionError:
        print(f"⚠️  File is locked (attempt {attempt+1}/3). Close cleaned_data.csv and retrying in 3s...")
        time.sleep(3)
else:
    raise PermissionError("Could not save — please close cleaned_data.csv and re-run.")
print(f"\n Saved cleaned data → {CLEAN_FILE}")
print(f"   Ready for Step 2: python step2_train_model.py")
