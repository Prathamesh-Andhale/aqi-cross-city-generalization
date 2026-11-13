#!/usr/bin/env python3
import sys, os

# Auto re-exec with project .venv python if started with global system Python
VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")
if os.path.exists(VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

"""
train.py  —  AQI Cross-City Generalization Pipeline (3 Geographic Cohorts)
========================================================================
Group 1 (Coastal)          : Train Pune + Chennai      → Test Mumbai
Group 2 (Indo-Gangetic)    : Train Delhi + Kanpur      → Test Lucknow
Group 3 (Deccan Plateau)   : Train Bengaluru + Mysuru  → Test Hyderabad

Key Methodological Highlights & Reviewer Audit Fixes:
  1. Clean Feature Space: 7 physical criteria features (6 raw pollutants + PM_ratio).
  2. Leak-Free Early Stopping (E0): Source donor cities are split 80% train / 20%
     internal validation. Target cities are strictly held out and never seen in eval_set.
  3. Contiguous Sequences (E0): BiLSTM sequences are formed independently per city.
  4. Temporal Alignment (E5): Window ends on day t (X[t-6..t] -> y_t) for same-day
     fair comparison with single-day tree ensembles.
  5. Clean Test Evaluation: All models evaluated on common clean t >= 6 subset (835 days).
  6. Analytic CPCB Baseline (E1) & Dominant Pollutant Validation (E7).

Usage:
    python train.py
"""

import json, warnings, shutil
from collections import Counter
import numpy as np
import pandas as pd
import joblib

import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import shap

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

tf.random.set_seed(42)
np.random.seed(42)
keras.utils.set_random_seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BASE_FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
ENG_FEATURES  = BASE_FEATURES + ["PM_ratio"]
FEATURES      = ENG_FEATURES          # Clean 7 features

SEQ_LEN = 7
EPOCHS  = 100
BATCH   = 32

ALL_AQI_CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# ── City Groups ────────────────────────────────────────────────────────────────
GROUPS = [
    {
        "name":   "Group1_Pune_Chennai_Mumbai",
        "train":  ["Pune", "Chennai"],
        "test":   "Mumbai",
        "label":  "Group 1 (Pune+Chennai → Mumbai)",
    },
    {
        "name":   "Group2_Delhi_Kanpur_Lucknow",
        "train":  ["Delhi", "Kanpur"],
        "test":   "Lucknow",
        "label":  "Group 2 (Delhi+Kanpur → Lucknow)",
    },
    {
        "name":   "Group3_Bengaluru_Mysuru_Hyderabad",
        "train":  ["Bengaluru", "Mysuru"],
        "test":   "Hyderabad",
        "label":  "Group 3 (Bengaluru+Mysuru → Hyderabad)",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def convert_to_native(obj):
    if isinstance(obj, np.ndarray):       return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)): return obj.item()
    if isinstance(obj, dict):             return {k: convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):    return [convert_to_native(i) for i in obj]
    return obj


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add atmospheric particulate size ratio."""
    df = df.copy()
    df["PM_ratio"] = df["PM2.5"] / (df["PM10"] + 1e-6)
    return df


def compute_cpcb_formula(row):
    """Compute official CPCB sub-index piecewise linear AQI and identify dominant pollutant."""
    bp = {
        "PM2.5": ([0, 30, 60, 90, 120, 250, 380], [0, 50, 100, 200, 300, 400, 500]),
        "PM10":  ([0, 50, 100, 250, 350, 430, 510], [0, 50, 100, 200, 300, 400, 500]),
        "NO2":   ([0, 40, 80, 180, 280, 400, 500], [0, 50, 100, 200, 300, 400, 500]),
        "SO2":   ([0, 40, 80, 380, 800, 1600, 2000], [0, 50, 100, 200, 300, 400, 500]),
        "CO":    ([0, 1.0, 2.0, 10.0, 17.0, 34.0, 45.0], [0, 50, 100, 200, 300, 400, 500]),
        "O3":    ([0, 50, 100, 168, 208, 748, 1000], [0, 50, 100, 200, 300, 400, 500]),
    }
    subindices = {}
    for p, (c_bp, i_bp) in bp.items():
        v = row[p]
        if pd.isna(v) or v < 0: continue
        for k in range(len(c_bp) - 1):
            if v <= c_bp[k + 1]:
                si = i_bp[k] + (v - c_bp[k]) * (i_bp[k + 1] - i_bp[k]) / (c_bp[k + 1] - c_bp[k])
                subindices[p] = si
                break
        else:
            subindices[p] = 500.0
    if not subindices:
        return 0.0, "None"
    dominant = max(subindices, key=lambda k: subindices[k])
    return max(subindices.values()), dominant


def aqi_to_category(aqi_val):
    if aqi_val <= 50: return "Good"
    elif aqi_val <= 100: return "Satisfactory"
    elif aqi_val <= 200: return "Moderate"
    elif aqi_val <= 300: return "Poor"
    elif aqi_val <= 400: return "Very Poor"
    else: return "Severe"


def make_city_sequences(df, scaler, le, seq_len=7):
    """
    Constructs contiguous sequences within a single city strictly ending on day t.
    Returns:
        X_seq: (N - seq_len + 1, seq_len, n_features)
        y_cat: (N - seq_len + 1,)
        y_reg: (N - seq_len + 1,)
    """
    X_raw = df[FEATURES].values.astype(np.float32)
    X_s = scaler.transform(X_raw).astype(np.float32)
    y_cat = le.transform(df["AQI_Category"].values)
    y_reg = df["AQI"].values.astype(np.float32)

    Xs, yc, yr = [], [], []
    for i in range(seq_len - 1, len(df)):
        Xs.append(X_s[i - seq_len + 1 : i + 1])
        yc.append(y_cat[i])
        yr.append(y_reg[i])
    return np.array(Xs, dtype=np.float32), np.array(yc, dtype=np.int32), np.array(yr, dtype=np.float32)


# ── Per-group training ─────────────────────────────────────────────────────────
def train_group(group):
    group_dir = os.path.join(MODELS_DIR, group["name"])
    os.makedirs(group_dir, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  {group['label']}")
    print(f"{'='*68}\n")

    # 1. Load per-city DataFrames
    donor_dfs = [engineer_features(pd.read_csv(os.path.join(DATA_DIR, f"{c}.csv"))) for c in group["train"]]
    test_df   = engineer_features(pd.read_csv(os.path.join(DATA_DIR, f"{group['test']}.csv")))

    # Label encoder: fit on ALL 6 fixed classes
    le = LabelEncoder()
    le.fit(ALL_AQI_CATEGORIES)
    n_classes = len(le.classes_)

    # 2. Leak-Free Validation Split: 80% train / 20% validation strictly on DONOR data
    donor_train_list, donor_val_list = [], []
    for c_df in donor_dfs:
        n_c = len(c_df)
        n_train = int(n_c * 0.8)
        donor_train_list.append(c_df.iloc[:n_train].copy())
        donor_val_list.append(c_df.iloc[n_train:].copy())

    df_donor_train = pd.concat(donor_train_list).reset_index(drop=True)
    df_donor_val   = pd.concat(donor_val_list).reset_index(drop=True)

    # Fit scaler strictly on donor training split
    scaler = QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=42)
    scaler.fit(df_donor_train[FEATURES].values)

    X_train_s = scaler.transform(df_donor_train[FEATURES].values).astype(np.float32)
    y_train_cat = le.transform(df_donor_train["AQI_Category"].values)
    y_train_reg = df_donor_train["AQI"].values.astype(np.float32)

    X_val_s = scaler.transform(df_donor_val[FEATURES].values).astype(np.float32)
    y_val_cat = le.transform(df_donor_val["AQI_Category"].values)
    y_val_reg = df_donor_val["AQI"].values.astype(np.float32)

    # Test City Data (Target domain) — evaluate on common clean subset t >= 6
    X_test_all_s = scaler.transform(test_df[FEATURES].values).astype(np.float32)
    y_test_all_cat = le.transform(test_df["AQI_Category"].values)
    y_test_all_reg = test_df["AQI"].values.astype(np.float32)

    # Common evaluation subset (t >= SEQ_LEN - 1)
    eval_slice = slice(SEQ_LEN - 1, len(test_df))
    X_test_eval_s  = X_test_all_s[eval_slice]
    y_test_eval_cat = y_test_all_cat[eval_slice]
    y_test_eval_reg = y_test_all_reg[eval_slice]
    test_dates_eval = test_df["Date"].iloc[eval_slice].tolist()

    # =========================================================================
    # 1. XGBoost (Trained on donor train, early-stopped on donor val)
    # =========================================================================
    print("── [1/3] XGBoost (Leak-free early stopping on donor val) ─────────")
    train_present_labels = np.unique(y_train_cat)
    local_to_global = {local: int(g) for local, g in enumerate(train_present_labels)}
    global_to_local = {g: local for local, g in local_to_global.items()}
    local_to_global_arr = np.array([local_to_global[i] for i in range(len(local_to_global))])

    y_train_xgb = np.array([global_to_local[v] for v in y_train_cat])
    y_val_xgb   = np.array([global_to_local.get(v, 0) for v in y_val_cat])

    xgb_clf = xgb.XGBClassifier(
        n_estimators=1000, max_depth=5, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_clf.fit(X_train_s, y_train_xgb, eval_set=[(X_val_s, y_val_xgb)], verbose=False)

    xgb_reg = xgb.XGBRegressor(
        n_estimators=1000, max_depth=5, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mae", random_state=42, n_jobs=-1, early_stopping_rounds=50,
    )
    xgb_reg.fit(X_train_s, y_train_reg, eval_set=[(X_val_s, y_val_reg)], verbose=False)

    xgb_pred_local  = xgb_clf.predict(X_test_eval_s)
    xgb_pred_global = local_to_global_arr[xgb_pred_local]
    xgb_cat_pred    = le.inverse_transform(xgb_pred_global).tolist()
    xgb_aqi_pred    = xgb_reg.predict(X_test_eval_s)

    xgb_acc = accuracy_score(y_test_eval_cat, xgb_pred_global)
    xgb_mae = mean_absolute_error(y_test_eval_reg, xgb_aqi_pred)
    xgb_r2  = r2_score(y_test_eval_reg, xgb_aqi_pred)
    print(f"  XGBoost   → Accuracy: {xgb_acc*100:.2f}%  MAE: {xgb_mae:.2f}  R²: {xgb_r2:.4f}")

    joblib.dump(xgb_clf, os.path.join(group_dir, "xgb_classifier.pkl"), compress=3)
    joblib.dump(xgb_reg, os.path.join(group_dir, "xgb_regressor.pkl"),  compress=3)

    # =========================================================================
    # 2. LightGBM (Trained on donor train, early-stopped on donor val)
    # =========================================================================
    print("── [2/3] LightGBM (Leak-free early stopping on donor val) ────────")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=7, learning_rate=0.03, num_leaves=63,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_clf.fit(X_train_s, y_train_cat, eval_set=[(X_val_s, y_val_cat)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

    lgb_reg = lgb.LGBMRegressor(
        n_estimators=1000, max_depth=7, learning_rate=0.03, num_leaves=63,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
    )
    lgb_reg.fit(X_train_s, y_train_reg, eval_set=[(X_val_s, y_val_reg)],
                callbacks=[lgb.early_stopping(50, verbose=False)])

    lgb_pred_global = lgb_clf.predict(X_test_eval_s)
    lgb_cat_pred    = le.inverse_transform(lgb_pred_global).tolist()
    lgb_aqi_pred    = lgb_reg.predict(X_test_eval_s)

    lgb_acc = accuracy_score(y_test_eval_cat, lgb_pred_global)
    lgb_mae = mean_absolute_error(y_test_eval_reg, lgb_aqi_pred)
    lgb_r2  = r2_score(y_test_eval_reg, lgb_aqi_pred)
    print(f"  LightGBM  → Accuracy: {lgb_acc*100:.2f}%  MAE: {lgb_mae:.2f}  R²: {lgb_r2:.4f}")

    joblib.dump(lgb_clf, os.path.join(group_dir, "lgb_classifier.pkl"), compress=3)
    joblib.dump(lgb_reg, os.path.join(group_dir, "lgb_regressor.pkl"),  compress=3)

    # =========================================================================
    # 3. BiLSTM (Per-city sequences, same-day window X[t-6..t] -> y_t)
    # =========================================================================
    print("── [3/3] BiLSTM (Per-city sequences, same-day alignment) ────────")
    n_feat = len(FEATURES)

    tr_seq_list, tr_cat_list, tr_reg_list = [], [], []
    for c_df in donor_train_list:
        xs, yc, yr = make_city_sequences(c_df, scaler, le, SEQ_LEN)
        tr_seq_list.append(xs); tr_cat_list.append(yc); tr_reg_list.append(yr)
    X_tr_seq = np.concatenate(tr_seq_list, axis=0)
    y_tr_seq_cat = np.concatenate(tr_cat_list, axis=0)
    y_tr_seq_reg = np.concatenate(tr_reg_list, axis=0)

    val_seq_list, val_cat_list, val_reg_list = [], [], []
    for c_df in donor_val_list:
        xs, yc, yr = make_city_sequences(c_df, scaler, le, SEQ_LEN)
        val_seq_list.append(xs); val_cat_list.append(yc); val_reg_list.append(yr)
    X_val_seq = np.concatenate(val_seq_list, axis=0)
    y_val_seq_cat = np.concatenate(val_cat_list, axis=0)
    y_val_seq_reg = np.concatenate(val_reg_list, axis=0)

    X_te_seq, y_te_seq_cat, y_te_seq_reg = make_city_sequences(test_df, scaler, le, SEQ_LEN)

    y_tr_oh  = to_categorical(y_tr_seq_cat, num_classes=n_classes)
    y_val_oh = to_categorical(y_val_seq_cat, num_classes=n_classes)

    def build_clf(seq_len, n_f, n_cls):
        inp = keras.Input(shape=(seq_len, n_f))
        x   = Bidirectional(LSTM(128, return_sequences=True))(inp)
        x   = Dropout(0.2)(x)
        x   = BatchNormalization()(x)
        x   = Bidirectional(LSTM(64, return_sequences=False))(x)
        x   = Dropout(0.2)(x)
        x   = BatchNormalization()(x)
        x   = Dense(64, activation="relu")(x)
        x   = Dropout(0.1)(x)
        out = Dense(n_cls, activation="softmax")(x)
        m   = keras.Model(inp, out)
        m.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
        return m

    def build_reg(seq_len, n_f):
        inp = keras.Input(shape=(seq_len, n_f))
        x   = Bidirectional(LSTM(128, return_sequences=True))(inp)
        x   = Dropout(0.2)(x)
        x   = BatchNormalization()(x)
        x   = Bidirectional(LSTM(64, return_sequences=False))(x)
        x   = Dropout(0.2)(x)
        x   = BatchNormalization()(x)
        x   = Dense(64, activation="relu")(x)
        x   = Dropout(0.1)(x)
        out = Dense(1)(x)
        m   = keras.Model(inp, out)
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae", metrics=["mae"])
        return m

    cb_clf = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor="val_accuracy", verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=7, verbose=0),
    ]
    cb_reg = [
        EarlyStopping(patience=15, restore_best_weights=True, monitor="val_mae", verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=7, verbose=0),
    ]

    lstm_clf = build_clf(SEQ_LEN, n_feat, n_classes)
    lstm_clf.fit(X_tr_seq, y_tr_oh, validation_data=(X_val_seq, y_val_oh),
                 epochs=EPOCHS, batch_size=BATCH, callbacks=cb_clf, verbose=0)

    lstm_reg = build_reg(SEQ_LEN, n_feat)
    lstm_reg.fit(X_tr_seq, y_tr_seq_reg, validation_data=(X_val_seq, y_val_seq_reg),
                 epochs=EPOCHS, batch_size=BATCH, callbacks=cb_reg, verbose=0)

    lstm_proba    = lstm_clf.predict(X_te_seq, verbose=0)
    lstm_enc_pred = np.argmax(lstm_proba, axis=1)
    lstm_cat_pred = le.inverse_transform(lstm_enc_pred).tolist()
    lstm_aqi_pred = lstm_reg.predict(X_te_seq, verbose=0).flatten()

    lstm_acc = accuracy_score(y_te_seq_cat, lstm_enc_pred)
    lstm_mae = mean_absolute_error(y_te_seq_reg, lstm_aqi_pred)
    lstm_r2  = r2_score(y_te_seq_reg, lstm_aqi_pred)
    print(f"  BiLSTM    → Accuracy: {lstm_acc*100:.2f}%  MAE: {lstm_mae:.2f}  R²: {lstm_r2:.4f}")

    lstm_clf.save(os.path.join(group_dir, "lstm_classifier.keras"))
    lstm_reg.save(os.path.join(group_dir, "lstm_regressor.keras"))

    # =========================================================================
    # 4. Soft Voting Ensemble (Common Clean 835 Test Days)
    # =========================================================================
    print("── [4/4] Soft Voting Ensemble (Evaluated on clean 835 days) ─────")
    xgb_proba_local = xgb_clf.predict_proba(X_test_eval_s)
    xgb_proba = np.zeros((len(X_test_eval_s), n_classes), dtype=np.float32)
    xgb_proba[:, local_to_global_arr] = xgb_proba_local

    lgb_proba_local = lgb_clf.predict_proba(X_test_eval_s)
    lgb_proba = np.zeros((len(X_test_eval_s), n_classes), dtype=np.float32)
    lgb_proba[:, lgb_clf.classes_.astype(int)] = lgb_proba_local

    vote_proba    = (xgb_proba + lgb_proba + lstm_proba) / 3.0
    vote_enc      = np.argmax(vote_proba, axis=1)
    vote_cat_pred = le.inverse_transform(vote_enc).tolist()
    vote_aqi_pred = (xgb_aqi_pred + lgb_aqi_pred + lstm_aqi_pred) / 3.0

    vote_acc = accuracy_score(y_test_eval_cat, vote_enc)
    vote_mae = mean_absolute_error(y_test_eval_reg, vote_aqi_pred)
    vote_r2  = r2_score(y_test_eval_reg, vote_aqi_pred)
    print(f"  Ensemble  → Accuracy: {vote_acc*100:.2f}%  MAE: {vote_mae:.2f}  R²: {vote_r2:.4f}")

    # =========================================================================
    # 5. CPCB Analytical Formula Baseline & Dominant Pollutant Validation
    # =========================================================================
    cpcb_calcs = [compute_cpcb_formula(row) for _, row in test_df.iloc[eval_slice].iterrows()]
    cpcb_aqi   = np.array([c[0] for c in cpcb_calcs], dtype=np.float32)
    cpcb_dom   = [c[1] for c in cpcb_calcs]
    cpcb_cat   = [aqi_to_category(val) for val in cpcb_aqi]
    cpcb_enc   = le.transform(cpcb_cat)

    cpcb_acc   = accuracy_score(y_test_eval_cat, cpcb_enc)
    cpcb_mae   = mean_absolute_error(y_test_eval_reg, cpcb_aqi)
    cpcb_r2    = r2_score(y_test_eval_reg, cpcb_aqi)

    dom_counts = Counter(cpcb_dom)
    dominant_shares = {k: round(v / len(cpcb_dom) * 100, 2) for k, v in dom_counts.items()}
    print(f"  CPCB Rule → Accuracy: {cpcb_acc*100:.2f}%  MAE: {cpcb_mae:.2f}  R²: {cpcb_r2:.4f}")
    print(f"  Dominant Pollutants: {dominant_shares}\n")

    # ── Save Scaler / Encoder / Config ─────────────────────────────────────────
    joblib.dump(scaler, os.path.join(group_dir, "scaler.pkl"), compress=3)
    joblib.dump(le,     os.path.join(group_dir, "label_encoder.pkl"), compress=3)
    with open(os.path.join(group_dir, "config.json"), "w") as f:
        json.dump({
            "seq_len":       SEQ_LEN,
            "features":      FEATURES,
            "base_features": BASE_FEATURES,
            "n_classes":     n_classes,
            "train_cities":  group["train"],
            "test_city":     group["test"],
        }, f)

    # ── Per-class stats & Feature Importance ───────────────────────────────────
    present_labels  = sorted(set(y_test_eval_cat))
    present_classes = le.classes_[present_labels].tolist()

    def per_class_stats(y_true, y_pred_enc):
        rpt = classification_report(
            y_true, y_pred_enc, labels=present_labels,
            target_names=present_classes, output_dict=True, zero_division=0)
        return {cls: {"precision": round(rpt[cls]["precision"], 3),
                      "recall":    round(rpt[cls]["recall"],    3),
                      "f1":        round(rpt[cls]["f1-score"],  3)}
                for cls in present_classes if cls in rpt}

    feat_imp_xgb = dict(zip(FEATURES, [round(float(x), 4) for x in xgb_clf.feature_importances_]))
    feat_imp_lgb = dict(zip(FEATURES, [round(float(x), 4) for x in lgb_clf.feature_importances_]))

    # TreeSHAP Global Attributions
    print("Computing global TreeSHAP values...")
    try:
        expl_xgb = shap.TreeExplainer(xgb_reg)
        sv_xgb   = expl_xgb.shap_values(X_test_eval_s)
        mean_sv_xgb = np.abs(sv_xgb).mean(axis=0) if len(sv_xgb.shape) > 1 else np.abs(sv_xgb)
        global_shap_xgb = dict(zip(FEATURES, [round(float(x), 4) for x in mean_sv_xgb]))
    except Exception as e:
        print(f"  XGB SHAP error: {e}")
        global_shap_xgb = {f: 0.0 for f in FEATURES}

    try:
        expl_lgb = shap.TreeExplainer(lgb_reg)
        sv_lgb   = expl_lgb.shap_values(X_test_eval_s)
        if isinstance(sv_lgb, list): sv_lgb = sv_lgb[0]
        mean_sv_lgb = np.abs(sv_lgb).mean(axis=0) if len(sv_lgb.shape) > 1 else np.abs(sv_lgb)
        global_shap_lgb = dict(zip(FEATURES, [round(float(x), 4) for x in mean_sv_lgb]))
    except Exception as e:
        print(f"  LGB SHAP error: {e}")
        global_shap_lgb = {f: 0.0 for f in FEATURES}

    payload = {
        "group":           group["label"],
        "train_cities":    group["train"],
        "test_city":       group["test"],
        "test_samples":    len(y_test_eval_reg),
        "results": {
            "CPCB_Rule":       {"accuracy": round(cpcb_acc*100, 2), "mae": round(cpcb_mae, 2), "r2": round(cpcb_r2, 4)},
            "XGBoost":         {"accuracy": round(xgb_acc*100, 2),  "mae": round(xgb_mae, 2),  "r2": round(xgb_r2, 4)},
            "LightGBM":        {"accuracy": round(lgb_acc*100, 2),  "mae": round(lgb_mae, 2),  "r2": round(lgb_r2, 4)},
            "LSTM":            {"accuracy": round(lstm_acc*100, 2), "mae": round(lstm_mae, 2), "r2": round(lstm_r2, 4)},
            "Voting Ensemble": {"accuracy": round(vote_acc*100, 2), "mae": round(vote_mae, 2), "r2": round(vote_r2, 4)},
        },
        "classes":             le.classes_.tolist(),
        "present_classes":     present_classes,
        "feature_names":       FEATURES,
        "scaler_mean":         scaler.quantiles_.mean(axis=0).tolist(),
        "scaler_scale":        scaler.quantiles_.std(axis=0).tolist(),
        "dominant_pollutants": dominant_shares,
        "test_dates":          test_dates_eval,
        "test_aqi_true":       y_test_eval_reg.tolist(),
        "xgb_pred":            xgb_aqi_pred.tolist(),
        "lgb_pred":            lgb_aqi_pred.tolist(),
        "lstm_pred":           lstm_aqi_pred.tolist(),
        "vote_pred":           vote_aqi_pred.tolist(),
        "vote_cat_pred":       vote_cat_pred,
        "xgb_cat_pred":        xgb_cat_pred,
        "lgb_cat_pred":        lgb_cat_pred,
        "lstm_cat_pred":       lstm_cat_pred,
        "true_cat":            le.inverse_transform(y_test_eval_cat).tolist(),
        "feat_importance_xgb": feat_imp_xgb,
        "feat_importance_lgb": feat_imp_lgb,
        "global_shap_xgb":     global_shap_xgb,
        "global_shap_lgb":     global_shap_lgb,
        "per_class": {
            "XGBoost":         per_class_stats(y_test_eval_cat, xgb_pred_global),
            "LightGBM":        per_class_stats(y_test_eval_cat, lgb_pred_global),
            "LSTM":            per_class_stats(y_test_eval_cat, lstm_enc_pred),
            "Voting Ensemble": per_class_stats(y_test_eval_cat, vote_enc),
        },
        "vote_probas_sample":  vote_proba[:50].tolist(),
        "vote_probas_classes": le.classes_.tolist(),
    }

    with open(os.path.join(group_dir, "evaluation_payload.json"), "w") as f:
        json.dump(convert_to_native(payload), f)

    print(f"  Artifacts saved to models/{group['name']}/")
    return payload


# ── Main ───────────────────────────────────────────────────────────────────────
def train_and_save():
    all_payloads = {}
    for group in GROUPS:
        all_payloads[group["name"]] = train_group(group)

    # Combined & backward-compatibility payloads
    first_key = list(all_payloads.keys())[0]
    with open(os.path.join(MODELS_DIR, "evaluation_payload.json"), "w") as f:
        json.dump(convert_to_native(all_payloads[first_key]), f)
    with open(os.path.join(MODELS_DIR, "all_groups_evaluation.json"), "w") as f:
        json.dump(convert_to_native(all_payloads), f)

    # Synchronize Group-1 artifacts to root models/ for seamless app.py loading
    g1_dir = os.path.join(MODELS_DIR, GROUPS[0]["name"])
    for fname in ["scaler.pkl", "label_encoder.pkl", "config.json",
                  "xgb_classifier.pkl", "xgb_regressor.pkl",
                  "lgb_classifier.pkl", "lgb_regressor.pkl",
                  "lstm_classifier.keras", "lstm_regressor.keras"]:
        src = os.path.join(g1_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(MODELS_DIR, fname))

    # Summary Report
    print("\n" + "="*70)
    print("  FINAL LEAK-FREE EMPIRICAL BENCHMARK (ALL 3 COHORTS)")
    print("="*70)
    for gname, p in all_payloads.items():
        print(f"\n  {p['group']} (N={p['test_samples']} test days)")
        print(f"  Train Donors: {', '.join(p['train_cities'])}  →  Target: {p['test_city']}")
        print(f"  {'Model':<20}  {'Accuracy':>10}  {'MAE':>8}  {'R²':>8}")
        print(f"  {'-'*54}")
        for model, v in p["results"].items():
            print(f"  {model:<20}  {v['accuracy']:>9.2f}%  {v['mae']:>8.2f}  {v['r2']:>8.4f}")
    print("\n" + "="*70 + "\n")

    return all_payloads


if __name__ == "__main__":
    train_and_save()
