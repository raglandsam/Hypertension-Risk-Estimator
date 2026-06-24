"""
Hypertension Risk Prediction - Training Pipeline v3
-----------------------------------------------------
Dataset: BRFSS 2015 health indicators (Diabetes/HighBP/etc.), real survey data,
balanced ~56/44 on HighBP. All columns are already numeric (binary flags 0/1,
or ordinal scales), so no categorical encoding is required this time.

Target: HighBP (= hypertension, "told by a health professional you have high
blood pressure" in the original BRFSS survey).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score, roc_auc_score
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_PATH = "diabetes_data.csv"
TARGET = "HighBP"
RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])
feature_columns = X.columns.tolist()

print("Class balance:\n", y.value_counts(normalize=True))

# Features that benefit from scaling (continuous-ish), rest are already 0/1 or
# small ordinal scales, scaling them too doesn't hurt linear models.
numeric_features = ["Age", "BMI", "GenHlth", "MentHlth", "PhysHlth"]

# ----------------------------------------------------------------------------
# 2. Split
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# ----------------------------------------------------------------------------
# 3. Train + compare
# ----------------------------------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
    "LinearSVC": LinearSVC(
        class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE
    ),
    "NaiveBayes": GaussianNB(),
}

if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
else:
    models["GradientBoosting"] = GradientBoostingClassifier(random_state=RANDOM_STATE)

best_name, best_f1, best_model = None, -1, None

print("\n" + "=" * 70)
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    report = classification_report(y_test, preds, target_names=["No HighBP", "HighBP"])
    macro_f1 = f1_score(y_test, preds, average="macro")

    try:
        if hasattr(model, "predict_proba"):
            auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        elif hasattr(model, "decision_function"):
            auc = roc_auc_score(y_test, model.decision_function(X_test_scaled))
        else:
            auc = None
    except Exception:
        auc = None

    print(f"\n--- {name} ---")
    print(report)
    print(f"Macro F1: {macro_f1:.4f}" + (f" | AUC: {auc:.4f}" if auc else ""))

    if macro_f1 > best_f1:
        best_f1, best_name, best_model = macro_f1, name, model

print("=" * 70)
print(f"\nBest model by macro F1: {best_name} (F1={best_f1:.4f})")

# ----------------------------------------------------------------------------
# 4. Feature importance (RandomForest, regardless of which model wins, since
#    it gives a stable, easy-to-interpret importance ranking)
# ----------------------------------------------------------------------------
importances = pd.Series(
    models["RandomForest"].feature_importances_, index=feature_columns
).sort_values(ascending=False)
print("\nFeature importances (RandomForest):")
print(importances)

# ----------------------------------------------------------------------------
# 5. Save model + metadata for the Streamlit app
# ----------------------------------------------------------------------------
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(
    {
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "best_model_name": best_name,
        "best_model_uses_decision_function": not hasattr(best_model, "predict_proba"),
        "data_means": X[numeric_features].mean().to_dict(),
        "data_stds": X[numeric_features].std().to_dict(),
        "data_mins": X[numeric_features].min().to_dict(),
        "data_maxs": X[numeric_features].max().to_dict(),
        "feature_importances": importances.to_dict(),
        "test_macro_f1": best_f1,
    },
    "model_metadata.pkl",
)

print("\nSaved: best_model.pkl, scaler.pkl, model_metadata.pkl")
print("Done.")
