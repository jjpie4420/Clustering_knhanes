# ============================================
# 32_prehypertension_foodgroup_ml.py
# ============================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    make_scorer
)

from xgboost import XGBClassifier

# --------------------------------------------
# PATH
# --------------------------------------------

INPUT = r"D:\precision_nutrition\FFQ\processed\analysis_cohort_semihealthy_foodgroups.csv"

OUT_DIR = r"D:\precision_nutrition\FFQ\results\32_prehypertension_foodgroup_ml"

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------
# LOAD
# --------------------------------------------

df = pd.read_csv(INPUT)

# --------------------------------------------
# PREHTN CLEAN ONLY
# --------------------------------------------

sub = df[df["bp_group"].isin([
    "normal_bp",
    "prehypertension_clean"
])].copy()

sub["target"] = np.where(
    sub["bp_group"] == "prehypertension_clean",
    1,
    0
)

print("[GROUP]")
print(sub["bp_group"].value_counts())

# --------------------------------------------
# FEATURES
# --------------------------------------------

food_features = [
    "fg_refined_grain_z",
    "fg_whole_grain_z",
    "fg_fastfood_z",
    "fg_meat_processed_z",
    "fg_fish_seafood_z",
    "fg_vegetable_z",
    "fg_kimchi_fermented_z",
    "fg_fruit_z",
    "fg_dairy_z",
    "fg_sweet_beverage_z",
    "fg_alcohol_z",
    "fg_coffee_tea_z"
]

X = sub[food_features].copy()
y = sub["target"].copy()

# --------------------------------------------
# CV
# --------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# --------------------------------------------
# MODELS
# --------------------------------------------

models = {

    "logistic_l2": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "xgboost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ))
    ])
}

# --------------------------------------------
# SCORING
# --------------------------------------------

scoring = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision"
}

# --------------------------------------------
# RUN
# --------------------------------------------

summary_rows = []

for model_name, model in models.items():

    print(f"\n[MODEL] {model_name}")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )

    row = {
        "model": model_name,
        "n": len(y),
        "prevalence": y.mean(),
        "n_features": X.shape[1],

        "auroc_mean": np.mean(scores["test_roc_auc"]),
        "auroc_sd": np.std(scores["test_roc_auc"]),

        "auprc_mean": np.mean(scores["test_average_precision"]),
        "auprc_sd": np.std(scores["test_average_precision"]),
    }

    summary_rows.append(row)

# --------------------------------------------
# SAVE SUMMARY
# --------------------------------------------

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    f"{OUT_DIR}/32_prehypertension_foodgroup_ml_summary.csv",
    index=False
)

print("\n[SUMMARY]")
print(summary)

# --------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------

# fit final xgb
final_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    ))
])

final_model.fit(X, y)

xgb_model = final_model.named_steps["model"]

importance = pd.DataFrame({
    "feature": food_features,
    "importance": xgb_model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

importance.to_csv(
    f"{OUT_DIR}/32_xgb_foodgroup_importance.csv",
    index=False
)

print("\n[XGB FEATURE IMPORTANCE]")
print(importance)

print("\n[SAVED]")
print(OUT_DIR)