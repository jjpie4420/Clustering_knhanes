# ============================================
# 36_sex_stratified_prehypertension_foodpattern.py
# ============================================

import os
import numpy as np
import pandas as pd

import shap

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate

from xgboost import XGBClassifier

# --------------------------------------------
# PATH
# --------------------------------------------

INPUT = r"D:\precision_nutrition\FFQ\processed\analysis_cohort_semihealthy_foodgroups.csv"

OUT_DIR = r"D:\precision_nutrition\FFQ\results\36_sex_stratified_prehypertension"

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------
# LOAD
# --------------------------------------------

df = pd.read_csv(INPUT)

# --------------------------------------------
# PREHTN CLEAN
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

feature_labels = {
    "fg_refined_grain_z": "Refined grain",
    "fg_whole_grain_z": "Whole grain",
    "fg_fastfood_z": "Fast food / sweets",
    "fg_meat_processed_z": "Meat / processed meat",
    "fg_fish_seafood_z": "Fish / seafood",
    "fg_vegetable_z": "Vegetables",
    "fg_kimchi_fermented_z": "Kimchi / fermented veg",
    "fg_fruit_z": "Fruit",
    "fg_dairy_z": "Dairy",
    "fg_sweet_beverage_z": "Sweet beverage",
    "fg_alcohol_z": "Alcohol",
    "fg_coffee_tea_z": "Coffee / tea"
}

# --------------------------------------------
# SEX LOOP
# --------------------------------------------

results = []

for sex_value, sex_label in zip([1, 2], ["Male", "Female"]):

    print(f"\n==============================")
    print(f"[SEX] {sex_label}")
    print(f"==============================")

    tmp = sub[sub["sex"] == sex_value].copy()

    print(tmp["bp_group"].value_counts())

    X = tmp[food_features].copy()
    y = tmp["target"].copy()

    # numeric
    for c in food_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # impute
    imp = SimpleImputer(strategy="median")

    X_imp = imp.fit_transform(X)

    X_imp = pd.DataFrame(
        X_imp,
        columns=food_features
    )

    # ----------------------------------------
    # MODEL
    # ----------------------------------------

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    # ----------------------------------------
    # CV PERFORMANCE
    # ----------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_validate(
        model,
        X_imp,
        y,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "average_precision": "average_precision"
        },
        n_jobs=-1
    )

    auroc = np.mean(scores["test_roc_auc"])
    auprc = np.mean(scores["test_average_precision"])

    print(f"\nAUROC={auroc:.3f}")
    print(f"AUPRC={auprc:.3f}")

    # ----------------------------------------
    # FIT FINAL
    # ----------------------------------------

    model.fit(X_imp, y)

    # ----------------------------------------
    # SHAP
    # ----------------------------------------

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_imp)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": food_features,
        "feature_label": [
            feature_labels[x]
            for x in food_features
        ],
        "mean_abs_shap": mean_abs_shap
    })

    shap_df = shap_df.sort_values(
        "mean_abs_shap",
        ascending=False
    )

    shap_df["sex"] = sex_label

    print("\n[TOP SHAP]")
    print(shap_df.head(10))

    shap_df.to_csv(
        f"{OUT_DIR}/{sex_label}_shap_summary.csv",
        index=False
    )

    results.append({
        "sex": sex_label,
        "n": len(tmp),
        "prevalence": y.mean(),
        "auroc": auroc,
        "auprc": auprc
    })

# --------------------------------------------
# SAVE PERFORMANCE
# --------------------------------------------

perf = pd.DataFrame(results)

perf.to_csv(
    f"{OUT_DIR}/36_sex_stratified_performance.csv",
    index=False
)

print("\n[PERFORMANCE]")
print(perf)

print("\n[SAVED]")
print(OUT_DIR)