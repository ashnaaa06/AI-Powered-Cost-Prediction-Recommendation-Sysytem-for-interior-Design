"""
STEP 2 - MODEL TRAINING
=======================
Run:  python step2_train_model.py

Models trained (13 base + 1 stacking = 14 total):
  Linear  : Ridge, Lasso, ElasticNet, BayesianRidge
  Bagging : Decision Tree, Random Forest
  Boosting: GradientBoosting, XGBoost, LightGBM, CatBoost
  Other   : KNN, SVR, MLP
  Ensemble: Stacking (top-4 base models + Ridge meta-learner)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.base            import clone
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler, PolynomialFeatures
from sklearn.ensemble        import (
    StackingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model    import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree            import DecisionTreeRegressor
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.svm             import SVR
from sklearn.neural_network  import MLPRegressor
from sklearn.metrics         import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error,
)
from xgboost  import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
CLEAN_FILE   = "data/cleaned_data.csv"
MODEL_DIR    = "model"
MODEL_FILE   = os.path.join(MODEL_DIR, "cost_model.pkl")
INFO_FILE    = os.path.join(MODEL_DIR, "model_info.json")
RANDOM_STATE = 42
TEST_SIZE    = 0.20
TARGET       = "total_cost_inr"
TARGET_R2    = 0.85   # if not reached, polynomial boost pass runs automatically

os.makedirs(MODEL_DIR, exist_ok=True)

SEP  = "=" * 65
DASH = "-" * 65

# =============================================================================
# LOAD & VALIDATE
# =============================================================================
print(SEP)
print("  STEP 2: MODEL TRAINING")
print(SEP)

df = pd.read_csv(CLEAN_FILE)
print(f"\n  Loaded  : {df.shape[0]} rows x {df.shape[1]} cols")

if df.select_dtypes(include="object").shape[1] > 0:
    raise ValueError("Non-numeric columns found. Re-run step1_clean_data.py.")

if df.isnull().sum().sum() > 0:
    raise ValueError("Null values found. Re-run step1_clean_data.py.")

print(f"  Checks  : all-numeric [OK]   zero-nulls [OK]")

# =============================================================================
# FEATURE ENGINEERING
# Gives models cross-column signals they cannot discover by splitting one
# column at a time. Each feature has a clear business reason.
# =============================================================================
print(f"\n{DASH}")
print("  FEATURE ENGINEERING")
print(DASH)

base_cols = set(df.columns)
c = df.columns.tolist()

def add(name, series, reason):
    df[name] = series
    print(f"  + {name:<32}  {reason}")

# Area x quality (main cost driver)
if "property_area_sqft" in c and "materials_grade" in c:
    add("area_x_grade",
        df["property_area_sqft"] * df["materials_grade"],
        "area * grade  [main cost driver]")

if "rooms" in c and "materials_grade" in c:
    add("rooms_x_grade",
        df["rooms"] * df["materials_grade"],
        "rooms * grade  [finish complexity]")

if all(x in c for x in ["property_area_sqft", "rooms", "materials_grade"]):
    add("avg_room_size_x_grade",
        (df["property_area_sqft"] / df["rooms"].clip(lower=1)) * df["materials_grade"],
        "avg room size * grade")

# Non-linear area transforms (help linear models)
if "property_area_sqft" in c:
    add("log_area",  np.log1p(df["property_area_sqft"]),        "log(1+area)")
    add("sqrt_area", np.sqrt(df["property_area_sqft"]),          "sqrt(area)")
    add("area_sq",   df["property_area_sqft"] ** 2 / 1_000_000, "area^2 / 1e6")

# Labour / duration complexity
if "estimated_labour_days" in c:
    add("log_labour", np.log1p(df["estimated_labour_days"]), "log(1+labour_days)")

if "project_duration_days" in c and "rooms" in c:
    add("days_per_room",
        df["project_duration_days"] / df["rooms"].clip(lower=1),
        "duration / rooms  [complexity]")

if "estimated_labour_days" in c and "property_area_sqft" in c:
    add("labour_per_100sqft",
        df["estimated_labour_days"] / df["property_area_sqft"].clip(lower=1) * 100,
        "labour days per 100 sqft")

# Material variety
if "n_materials" in c and "rooms" in c:
    add("materials_per_room",
        df["n_materials"] / df["rooms"].clip(lower=1),
        "material types per room")

if "n_materials" in c and "materials_grade" in c:
    add("materials_x_grade",
        df["n_materials"] * df["materials_grade"],
        "material count * grade  [material spend]")

# Furniture luxury signals
if "furniture_included" in c and "materials_grade" in c:
    add("furniture_x_grade",
        df["furniture_included"] * df["materials_grade"],
        "furnished * grade  [luxury flag]")

if "furniture_included" in c and "property_area_sqft" in c:
    add("furniture_x_area",
        df["furniture_included"] * df["property_area_sqft"],
        "furnished * area")

# Full-scope flag (drop_first in step1 OHE makes full-scope the implicit base)
scope_dummies = [col for col in df.columns if col.startswith("scope_")]
if scope_dummies and "property_area_sqft" in c:
    df["is_full_scope"] = (df[scope_dummies].sum(axis=1) == 0).astype(int)
    add("full_scope_x_area",
        df["is_full_scope"] * df["property_area_sqft"],
        "full-scope * area  [large full projects]")

# City tier x grade
tier_dummies = [col for col in df.columns if col.startswith("city_tier_")]
if tier_dummies and "materials_grade" in c:
    df["is_metro"] = (df[tier_dummies].sum(axis=1) == 0).astype(int)
    add("metro_x_grade",
        df["is_metro"] * df["materials_grade"],
        "metro city * grade  [premium metro cost]")

# Seasonality
if "project_quarter" in c and "property_area_sqft" in c:
    add("quarter_x_area",
        df["project_quarter"] * df["property_area_sqft"],
        "quarter * area  [seasonal large projects]")

print(f"\n  Features before : {len(base_cols) - 1}  (excl. target)")
print(f"  Features added  : {df.shape[1] - len(base_cols)}")
print(f"  Features total  : {df.shape[1] - 1}  (excl. target)")

# =============================================================================
# PREPARE X, y
# =============================================================================
FEAT_COLS = [col for col in df.columns if col != TARGET]
X         = df[FEAT_COLS].values.astype(np.float64)
y_raw     = df[TARGET].values.astype(np.float64)

# Log-transform target: spans a 100x range (right-skewed).
# np.expm1 reverses it when computing metrics.
y = np.log1p(y_raw)

skew_raw = float(pd.Series(y_raw).skew())
skew_log = float(pd.Series(y).skew())
print(f"\n  Target skewness : {skew_raw:.2f} (raw)  ->  {skew_log:.2f} (log)  [OK]")

X_train, X_test, y_train, y_test_log = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
_, _, _, y_test_raw = train_test_split(
    X, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"  Train / Test    : {X_train.shape[0]} / {X_test.shape[0]}")

# =============================================================================
# HELPERS
# =============================================================================
def make_pipe(estimator):
    """StandardScaler + estimator. Each model gets its own scaler instance."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def evaluate(pipe, X_te, y_te_raw):
    """Predict on log scale, inverse-transform, return metrics on Rs scale."""
    y_pred = np.maximum(np.expm1(pipe.predict(X_te)), 0)
    return {
        "r2"    : r2_score(y_te_raw, y_pred),
        "mae"   : mean_absolute_error(y_te_raw, y_pred),
        "rmse"  : float(np.sqrt(mean_squared_error(y_te_raw, y_pred))),
        "mape"  : mean_absolute_percentage_error(y_te_raw, y_pred) * 100,
    }

# =============================================================================
# MODEL DEFINITIONS  (13 models)
# =============================================================================
def build_models():
    return {
        # --- LINEAR ---
        "01 Ridge":
            make_pipe(Ridge(alpha=10)),

        "02 Lasso":
            make_pipe(Lasso(alpha=0.001, max_iter=5000)),

        "03 ElasticNet":
            make_pipe(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)),

        "04 BayesianRidge":
            make_pipe(BayesianRidge()),

        # --- BAGGING ---
        "05 Decision Tree":
            make_pipe(DecisionTreeRegressor(
                max_depth=12, min_samples_leaf=4,
                min_samples_split=8, random_state=RANDOM_STATE)),

        "06 Random Forest":
            make_pipe(RandomForestRegressor(
                n_estimators=500, max_depth=None,
                min_samples_leaf=2, max_features=0.6,
                n_jobs=-1, random_state=RANDOM_STATE)),

        # --- BOOSTING ---
        "07 GradientBoosting":
            make_pipe(GradientBoostingRegressor(
                n_estimators=1000, learning_rate=0.02,
                max_depth=4,       subsample=0.75,
                max_features=0.7,  min_samples_leaf=4,
                loss="squared_error", random_state=RANDOM_STATE)),

        "08 XGBoost":
            make_pipe(XGBRegressor(
                n_estimators=1000, learning_rate=0.02,
                max_depth=4,          subsample=0.75,
                colsample_bytree=0.7, colsample_bylevel=0.7,
                reg_alpha=0.05,       reg_lambda=1.5,
                min_child_weight=4,   gamma=0.1,
                tree_method="hist",   device="cpu",
                random_state=RANDOM_STATE, verbosity=0)),

        "09 LightGBM":
            make_pipe(LGBMRegressor(
                n_estimators=1000, learning_rate=0.02,
                max_depth=5,       num_leaves=40,
                subsample=0.75,    subsample_freq=1,
                colsample_bytree=0.7,
                reg_alpha=0.05,    reg_lambda=1.5,
                min_child_samples=25,
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)),

        "10 CatBoost":
            make_pipe(CatBoostRegressor(
                iterations=1000,  learning_rate=0.02,
                depth=5,          l2_leaf_reg=3,
                bagging_temperature=0.5,
                loss_function="RMSE",
                random_seed=RANDOM_STATE, verbose=False)),

        # --- OTHER ---
        "11 KNN":
            make_pipe(KNeighborsRegressor(
                n_neighbors=10, weights="distance", n_jobs=-1)),

        "12 SVR":
            make_pipe(SVR(kernel="rbf", C=10, epsilon=0.05, gamma="scale")),

        "13 MLP":
            make_pipe(MLPRegressor(
                hidden_layer_sizes=(512, 256, 128, 64),
                activation="relu",   solver="adam",
                learning_rate="adaptive", learning_rate_init=0.001,
                max_iter=1000,        early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20,
                random_state=RANDOM_STATE)),
    }

# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_all(models_dict, X_tr, y_tr, X_te, y_te_raw):
    results = {}
    t_start = time.time()

    print(f"\n{SEP}")
    print(f"  TRAINING {len(models_dict)} BASE MODELS")
    print(SEP)

    for name, pipe in models_dict.items():
        print(f"  [{name}]", end="  ", flush=True)
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        metrics = evaluate(pipe, X_te, y_te_raw)
        elapsed = time.time() - t0
        metrics["time_s"] = round(elapsed, 1)
        metrics["pipe"]   = pipe
        results[name]     = metrics
        print(f"R2={metrics['r2']:.4f}  MAPE={metrics['mape']:.2f}%  ({elapsed:.1f}s)")

    print(f"\n  Base training time: {time.time()-t_start:.0f}s")
    return results


# =============================================================================
# FIRST PASS
# =============================================================================
models  = build_models()
results = train_all(models, X_train, y_train, X_test, y_test_raw)

# =============================================================================
# AUTO ACCURACY BOOST
# If best R2 < TARGET_R2, add polynomial interaction features on top-5
# numeric columns and retrain the 3 best boosting models.
# =============================================================================
best_r2_pass1 = max(v["r2"] for v in results.values())
print(f"\n  Best R2 after pass 1: {best_r2_pass1:.4f}  (target >= {TARGET_R2})")

if best_r2_pass1 < TARGET_R2:
    print(f"\n  R2 < {TARGET_R2}. Running polynomial boost pass...")

    poly_cols = []
    for col in ["area_x_grade", "log_area", "rooms_x_grade",
                "avg_room_size_x_grade", "materials_x_grade",
                "property_area_sqft", "materials_grade", "rooms"]:
        if col in FEAT_COLS:
            poly_cols.append(col)
        if len(poly_cols) == 5:
            break

    if poly_cols:
        poly_idx     = [FEAT_COLS.index(c) for c in poly_cols]
        pf           = PolynomialFeatures(degree=2, include_bias=False,
                                          interaction_only=False)
        X_poly_extra = pf.fit_transform(X[:, poly_idx])
        X_aug        = np.hstack([X, X_poly_extra[:, len(poly_cols):]])

        X_tr_aug, X_te_aug, y_tr_aug, _ = train_test_split(
            X_aug, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        _, _, _, y_te_raw_aug = train_test_split(
            X_aug, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        print(f"  Augmented feature count: {X_aug.shape[1]}")
        print(f"\n{DASH}")
        print(f"  BOOST PASS - 3 boosting models on polynomial features")
        print(DASH)

        boost_models = {
            "B1 XGBoost-poly": make_pipe(XGBRegressor(
                n_estimators=1000, learning_rate=0.02, max_depth=4,
                subsample=0.75, colsample_bytree=0.7,
                reg_alpha=0.05, reg_lambda=1.5, min_child_weight=4,
                tree_method="hist", device="cpu",
                random_state=RANDOM_STATE, verbosity=0)),

            "B2 LightGBM-poly": make_pipe(LGBMRegressor(
                n_estimators=1000, learning_rate=0.02, max_depth=5,
                num_leaves=40, subsample=0.75, colsample_bytree=0.7,
                reg_alpha=0.05, reg_lambda=1.5,
                n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)),

            "B3 GBM-poly": make_pipe(GradientBoostingRegressor(
                n_estimators=1000, learning_rate=0.02, max_depth=4,
                subsample=0.75, max_features=0.7, min_samples_leaf=4,
                loss="squared_error", random_state=RANDOM_STATE)),
        }

        for bname, bpipe in boost_models.items():
            print(f"  [{bname}]", end="  ", flush=True)
            t0 = time.time()
            bpipe.fit(X_tr_aug, y_tr_aug)
            bmet = evaluate(bpipe, X_te_aug, y_te_raw_aug)
            elapsed = time.time() - t0
            bmet["time_s"] = round(elapsed, 1)
            bmet["pipe"]   = bpipe
            results[bname] = bmet
            print(f"R2={bmet['r2']:.4f}  MAPE={bmet['mape']:.2f}%  ({elapsed:.1f}s)")

# =============================================================================
# STACKING ENSEMBLE  (top-4 by R2 + Ridge meta-learner)
# =============================================================================
print(f"\n{DASH}")
print("  STACKING ENSEMBLE  (top-4 base models + Ridge meta-learner)")
print(DASH)

ranked_base = sorted(results.items(), key=lambda x: x[1]["r2"], reverse=True)
top4_names  = [n for n, _ in ranked_base[:4]]
print(f"  Top-4 selected: {[n[3:] for n in top4_names]}")

stack_estimators = [
    (n[3:].replace(" ", "_").replace("-", "_"), clone(results[n]["pipe"]))
    for n in top4_names
]
stack_model = StackingRegressor(
    estimators      = stack_estimators,
    final_estimator = Ridge(alpha=1.0),
    cv              = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    passthrough     = False,
    n_jobs          = -1,
)

print("  Training stacking...", end="  ", flush=True)
t0 = time.time()
stack_model.fit(X_train, y_train)
elapsed = time.time() - t0

stack_m = evaluate(stack_model, X_test, y_test_raw)
stack_m["time_s"] = round(elapsed, 1)
stack_m["pipe"]   = stack_model
results["ST Stacking"] = stack_m
print(f"R2={stack_m['r2']:.4f}  MAPE={stack_m['mape']:.2f}%  ({elapsed:.1f}s)")

# =============================================================================
# RESULTS TABLE
# =============================================================================
ranked = sorted(results.items(), key=lambda x: x[1]["r2"], reverse=True)

print(f"\n\n{SEP}")
print(f"  RESULTS  ({len(results)} models, ranked by R2)")
print(SEP)
print(f"  {'#':<4}  {'Model':<24}  {'MAE':>12}  {'RMSE':>12}  "
      f"{'R2':>7}  {'MAPE%':>7}  {'Time':>6}")
print(f"  {'-' * 75}")

for i, (name, m) in enumerate(ranked, 1):
    flag  = "  <<< BEST" if i == 1 else ""
    label = name[3:] if len(name) > 3 else name
    print(
        f"  {i:>2}.  {label:<24}"
        f"  {m['mae']:>11,.0f}"
        f"  {m['rmse']:>11,.0f}"
        f"  {m['r2']:>7.4f}"
        f"  {m['mape']:>6.2f}%"
        f"  {m['time_s']:>5.1f}s"
        f"{flag}"
    )

print(f"  {'-' * 75}")
print()
print("  R2   - fraction of variance explained  (higher is better)")
print("  MAPE - average % error per prediction  (lower is better)")
print("  MAE  - average absolute rupee error    (lower is better)")
print("  All metrics on original Rs scale via np.expm1(prediction)")

# =============================================================================
# BEST MODEL
# =============================================================================
best_name = ranked[0][0]
best_m    = results[best_name]
best_pipe = best_m["pipe"]

print(f"\n{SEP}")
print(f"  BEST MODEL : {best_name[3:]}")
print(f"  R2         : {best_m['r2']:.4f}  ({best_m['r2']*100:.1f}% variance explained)")
print(f"  MAE        : Rs {best_m['mae']:,.0f}")
print(f"  RMSE       : Rs {best_m['rmse']:,.0f}")
print(f"  MAPE       : {best_m['mape']:.2f}%")
if best_m["r2"] >= TARGET_R2:
    print(f"  Target R2 >= {TARGET_R2} : ACHIEVED")
else:
    print(f"  Target R2 >= {TARGET_R2} : not reached (dataset may have limited signal)")
print(SEP)

# =============================================================================
# FEATURE IMPORTANCES  (tree-based models support .feature_importances_)
# =============================================================================
TREE_MODELS = {
    "05 Decision Tree", "06 Random Forest",
    "07 GradientBoosting", "08 XGBoost", "09 LightGBM", "10 CatBoost",
    "B1 XGBoost-poly", "B2 LightGBM-poly", "B3 GBM-poly",
}

importances = []
imp_label   = ""

if best_name in TREE_MODELS:
    try:
        imps        = best_pipe.named_steps["model"].feature_importances_
        importances = sorted(zip(FEAT_COLS, imps), key=lambda x: -x[1])
        imp_label   = f"{best_name[3:]} importances"
    except Exception:
        pass

if not importances:
    # Fallback: always available
    imps        = results["06 Random Forest"]["pipe"].named_steps["model"].feature_importances_
    importances = sorted(zip(FEAT_COLS, imps), key=lambda x: -x[1])
    imp_label   = "Random Forest importances (reference)"

print(f"\n  TOP 15 FEATURE IMPORTANCES  ({imp_label})")
print(f"  {'-' * 58}")
max_imp = importances[0][1] if importances else 1.0
for feat, imp in importances[:15]:
    bar = "#" * int(35 * imp / max_imp)
    print(f"  {feat:<34}  {imp:.5f}  {bar}")

# =============================================================================
# CROSS-VALIDATION  (5-fold on best model)
# =============================================================================
print(f"\n  5-Fold Cross-Validation  ({best_name[3:]})")
cv_scores = cross_val_score(clone(best_pipe), X, y, cv=5, scoring="r2", n_jobs=-1)
print(f"  Fold R2   : {[round(s, 4) for s in cv_scores]}")
print(f"  Mean R2   : {cv_scores.mean():.4f}  +/-  {cv_scores.std():.4f}")

# =============================================================================
# OVERFIT CHECK
# =============================================================================
X_tr2, _, y_tr_raw2, _ = train_test_split(
    X, y_raw, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
train_pred = np.maximum(np.expm1(best_pipe.predict(X_tr2)), 0)
train_r2   = r2_score(y_tr_raw2, train_pred)
gap        = train_r2 - best_m["r2"]
verdict    = "WARNING - possible overfit, consider more regularisation" if gap > 0.06 else "OK"

print(f"\n  Overfit check:")
print(f"  Train R2  {train_r2:.4f}   Test R2  {best_m['r2']:.4f}   "
      f"Gap  {gap:.4f}   [{verdict}]")

# =============================================================================
# SAVE MODEL
# =============================================================================
ts      = datetime.now().strftime("%Y%m%d_%H%M")
ts_file = MODEL_FILE.replace(".pkl", f"_{ts}.pkl")
joblib.dump(best_pipe, MODEL_FILE)
joblib.dump(best_pipe, ts_file)

print(f"\n  Saved model  -> {MODEL_FILE}")
print(f"  Saved backup -> {ts_file}")
print(f"  Inference    : np.maximum(np.expm1(model.predict(X_input)), 0)")

# =============================================================================
# SAVE model_info.json
# =============================================================================
model_info = {
    "best_model"        : best_name[3:],
    "log_target"        : True,
    "inverse_transform" : "np.expm1",
    "r2_score"          : round(best_m["r2"],   4),
    "mae"               : round(best_m["mae"],  2),
    "rmse"              : round(best_m["rmse"], 2),
    "mape"              : round(best_m["mape"], 4),
    "train_r2"          : round(train_r2,       4),
    "cv_mean_r2"        : round(float(cv_scores.mean()), 4),
    "cv_std_r2"         : round(float(cv_scores.std()),  4),
    "feature_cols"      : FEAT_COLS,
    "n_features"        : len(FEAT_COLS),
    "n_models_compared" : len(results),
    "target_r2"         : TARGET_R2,
    "target_achieved"   : best_m["r2"] >= TARGET_R2,
    "all_results"       : {
        n[3:]: {k: round(v, 4) if isinstance(v, float) else v
                for k, v in m.items() if k != "pipe"}
        for n, m in results.items()
    },
    "model_ranking"     : [
        {"rank": i+1, "model": n[3:],
         "r2"  : round(m["r2"],  4),
         "mae" : round(m["mae"], 2),
         "mape": round(m["mape"], 4)}
        for i, (n, m) in enumerate(ranked)
    ],
    "top_features"      : [
        {"feature": feat, "importance": round(float(imp), 6)}
        for feat, imp in importances[:15]
    ],
    "timestamp"         : ts,
}

with open(INFO_FILE, "w") as fh:
    json.dump(model_info, fh, indent=2)
print(f"  Saved info   -> {INFO_FILE}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{SEP}")
print(f"  COMPLETE  -  {len(results)} models trained")
print(SEP)
print(f"  Best model : {best_name[3:]}")
print(f"  R2         : {best_m['r2']:.4f}  ({best_m['r2']*100:.1f}%)")
print(f"  MAE        : Rs {best_m['mae']:,.0f}")
print(f"  MAPE       : {best_m['mape']:.2f}%")
print(f"  CV R2      : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"  Next step  : python step3_app.py")
print(f"  Inference  : np.maximum(np.expm1(model.predict(X)), 0)")
print(SEP)