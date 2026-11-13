#!/usr/bin/env python3
import sys, os

# Auto re-exec with project .venv python if started with global system Python
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if os.path.exists(VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

"""
run_reduced_sensors.py — Reviewer Experiment E9: Hardware Degradation Benchmark
==============================================================================
Evaluates cross-city transfer accuracy across 4 realistic sensor deployment tiers:
  - Tier 1: Complete CAAQMS (PM2.5, PM10, NO2, SO2, CO, O3, PM_ratio)
  - Tier 2: Particulate-Only Low-Cost Station (PM2.5, PM10, PM_ratio)
  - Tier 3: Particulate + Traffic/Combustion (PM2.5, PM10, PM_ratio, NO2)
  - Tier 4: Minimal Single Particulate Sensor (PM2.5)

Evaluated across all 3 geographic cohorts:
  - Cohort 1: Pune + Chennai  → Mumbai (Target)
  - Cohort 2: Delhi + Kanpur  → Lucknow (Target)
  - Cohort 3: Bengaluru + Mysuru → Hyderabad (Target)

Outputs:
  - models/reduced_sensor_evaluation.json
"""

import json, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

GROUPS = [
    {"name": "Cohort1_Mumbai",    "train": ["Pune", "Chennai"],       "test": "Mumbai"},
    {"name": "Cohort2_Lucknow",   "train": ["Delhi", "Kanpur"],       "test": "Lucknow"},
    {"name": "Cohort3_Hyderabad", "train": ["Bengaluru", "Mysuru"],   "test": "Hyderabad"},
]

TIERS = {
    "Tier1_Full_6Pollutants":       ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "PM_ratio"],
    "Tier2_Particulate_Only":       ["PM2.5", "PM10", "PM_ratio"],
    "Tier3_Particulate_Combustion": ["PM2.5", "PM10", "PM_ratio", "NO2"],
    "Tier4_Single_Particulate":     ["PM2.5"],
}

ALL_AQI_CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def engineer(df):
    df = df.copy()
    df["PM_ratio"] = df["PM2.5"] / (df["PM10"] + 1e-6)
    return df


def run_benchmark():
    le = LabelEncoder()
    le.fit(ALL_AQI_CATEGORIES)

    results = {}

    print("\n" + "="*74)
    print("  REVIEWER EXPERIMENT E9: REDUCED-SENSOR HARDWARE DEGRADATION BENCHMARK")
    print("="*74)

    for g in GROUPS:
        gname = g["name"]
        print(f"\n--- {gname} (Train: {', '.join(g['train'])} -> Test: {g['test']}) ---")
        results[gname] = {"train": g["train"], "test": g["test"], "tiers": {}}

        donor_dfs = [engineer(pd.read_csv(os.path.join(DATA_DIR, f"{c}.csv"))) for c in g["train"]]
        test_df   = engineer(pd.read_csv(os.path.join(DATA_DIR, f"{g['test']}.csv")))

        # 80/20 internal donor split for leak-free early stopping
        donor_train_list, donor_val_list = [], []
        for c_df in donor_dfs:
            n_c = len(c_df)
            n_train = int(n_c * 0.8)
            donor_train_list.append(c_df.iloc[:n_train].copy())
            donor_val_list.append(c_df.iloc[n_train:].copy())

        df_train = pd.concat(donor_train_list).reset_index(drop=True)
        df_val   = pd.concat(donor_val_list).reset_index(drop=True)

        y_tr_cat = le.transform(df_train["AQI_Category"].values)
        y_tr_reg = df_train["AQI"].values.astype(np.float32)
        y_val_cat = le.transform(df_val["AQI_Category"].values)
        y_val_reg = df_val["AQI"].values.astype(np.float32)

        # Common clean test evaluation (t >= 6, 835 test days)
        eval_slice = slice(6, len(test_df))
        df_test_eval = test_df.iloc[eval_slice]
        y_te_cat = le.transform(df_test_eval["AQI_Category"].values)
        y_te_reg = df_test_eval["AQI"].values.astype(np.float32)

        for tier_name, feats in TIERS.items():
            scaler = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=42)
            X_tr_s = scaler.fit_transform(df_train[feats].values).astype(np.float32)
            X_val_s = scaler.transform(df_val[feats].values).astype(np.float32)
            X_te_s  = scaler.transform(df_test_eval[feats].values).astype(np.float32)

            # LightGBM Classifier & Regressor
            clf = lgb.LGBMClassifier(
                n_estimators=500, max_depth=7, learning_rate=0.03, num_leaves=63,
                subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
                random_state=42, n_jobs=-1, verbose=-1
            )
            clf.fit(X_tr_s, y_tr_cat, eval_set=[(X_val_s, y_val_cat)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

            reg = lgb.LGBMRegressor(
                n_estimators=500, max_depth=7, learning_rate=0.03, num_leaves=63,
                subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
                random_state=42, n_jobs=-1, verbose=-1
            )
            reg.fit(X_tr_s, y_tr_reg, eval_set=[(X_val_s, y_val_reg)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

            pred_cat = clf.predict(X_te_s)
            pred_reg = reg.predict(X_te_s)

            acc = accuracy_score(y_te_cat, pred_cat)
            mae = mean_absolute_error(y_te_reg, pred_reg)
            r2  = r2_score(y_te_reg, pred_reg)

            tier_res = {
                "features": feats,
                "n_features": len(feats),
                "accuracy": round(acc * 100, 2),
                "mae": round(mae, 2),
                "r2": round(r2, 4),
            }
            results[gname]["tiers"][tier_name] = tier_res
            print(f"  {tier_name:<30} (k={len(feats)}) -> Acc: {acc*100:6.2f}% | MAE: {mae:5.2f} | R²: {r2:6.4f}")

    out_file = os.path.join(MODELS_DIR, "reduced_sensor_evaluation.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*74)
    print(f"  Saved benchmark to {out_file}")
    print("="*74 + "\n")
    return results


if __name__ == "__main__":
    run_benchmark()

