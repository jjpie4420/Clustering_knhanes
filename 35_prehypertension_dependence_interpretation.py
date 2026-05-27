# ============================================
# 35_prehypertension_dependence_interpretation.py
# ============================================

import os
import numpy as np
import pandas as pd

import shap
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

# --------------------------------------------
# PATH
# --------------------------------------------

INPUT = r"D:\precision_nutrition\FFQ\processed\analysis_cohort_semihealthy_foodgroups.csv"

OUT_DIR = r"D:\precision_nutrition\FFQ\results\35_prehypertension_dependence"

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
# DATA
# --------------------------------------------

X = sub[food_features].copy()
y = sub["target"].copy()

for c in food_features:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# impute
imp = SimpleImputer(strategy="median")

X_imp = imp.fit_transform(X)

X_imp = pd.DataFrame(
    X_imp,
    columns=food_features
)

# --------------------------------------------
# MODEL
# --------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_imp, y)

# --------------------------------------------
# SHAP
# --------------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_imp)

# --------------------------------------------
# TOP FEATURES
# --------------------------------------------

mean_abs_shap = np.abs(shap_values).mean(axis=0)

summary = pd.DataFrame({
    "feature": food_features,
    "importance": mean_abs_shap
})

summary = summary.sort_values(
    "importance",
    ascending=False
)

top_features = summary["feature"].head(5).tolist()

print("[TOP FEATURES]")
print(summary.head(5))

# --------------------------------------------
# DEPENDENCE PLOTS
# --------------------------------------------

for feat in top_features:

    print(f"\n[PLOT] {feat}")

    plt.figure(figsize=(6, 5))

    shap.dependence_plot(
        feat,
        shap_values,
        X_imp,
        interaction_index=None,
        feature_names=food_features,
        show=False
    )

    plt.xlabel(
        feature_labels.get(feat, feat)
    )

    plt.tight_layout()

    save_path = f"{OUT_DIR}/{feat}_dependence.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# --------------------------------------------
# SIMPLE QUANTILE SUMMARY
# --------------------------------------------

rows = []

for feat in top_features:

    tmp = pd.DataFrame({
        "x": X_imp[feat],
        "shap": shap_values[:, food_features.index(feat)]
    })

    tmp["quartile"] = pd.qcut(
        tmp["x"],
        q=4,
        labels=False,
        duplicates="drop"
    )

    q_summary = tmp.groupby("quartile").agg({
        "x": "mean",
        "shap": "mean"
    }).reset_index()

    q_summary["feature"] = feat

    rows.append(q_summary)

quantile_summary = pd.concat(rows)

quantile_summary.to_csv(
    f"{OUT_DIR}/35_quantile_shap_summary.csv",
    index=False
)

print("\n[QUANTILE SHAP SUMMARY]")
print(quantile_summary)

print("\n[SAVED]")
print(OUT_DIR)