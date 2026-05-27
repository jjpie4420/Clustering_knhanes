# 49_predm_age_stratified.py

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from xgboost import XGBClassifier
import shap

# =========================================================
# LOAD
# =========================================================

BASE_DIR = Path(r"D:\precision_nutrition\FFQ")
OUT_DIR = BASE_DIR / "results" / "49_predm_age_stratified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    BASE_DIR / "processed" / "analysis_cohort_semihealthy_foodgroups.csv",
    encoding="utf-8-sig",
    low_memory=False
)

# =========================================================
# GROUP DEFINE
# =========================================================

df = df[
    df["label_prediabetes_clean2"].isin([0,1])
].copy()

df["glucose_group"] = np.where(
    df["label_prediabetes_clean2"] == 1,
    "prediabetes_clean",
    "normal_glucose"
)

# =========================================================
# AGE GROUP
# =========================================================

df = df[df["age"].between(20, 64)]

df["age_group"] = pd.cut(
    df["age"],
    bins=[20, 40, 55, 65],
    labels=["20-39", "40-54", "55-64"],
    right=False
)

# =========================================================
# FEATURES
# =========================================================

food_features = [
    c for c in df.columns
    if c.startswith("fg_") and c.endswith("_z")
]

label_map = {
    "fg_refined_grain_z":"Refined grain",
    "fg_whole_grain_z":"Whole grain",
    "fg_fastfood_z":"Fast food / sweets",
    "fg_meat_processed_z":"Meat / processed meat",
    "fg_fish_seafood_z":"Fish / seafood",
    "fg_vegetable_z":"Vegetables",
    "fg_kimchi_fermented_z":"Kimchi / fermented veg",
    "fg_fruit_z":"Fruit",
    "fg_dairy_z":"Dairy",
    "fg_sweet_beverage_z":"Sweet beverage",
    "fg_alcohol_z":"Alcohol",
    "fg_coffee_tea_z":"Coffee / tea"
}

# =========================================================
# LOOP
# =========================================================

results = []

for age_group in df["age_group"].dropna().unique():

    print("\n==============================")
    print(f"[AGE] {age_group}")
    print("==============================")

    sub = df[df["age_group"] == age_group].copy()

    print(sub["glucose_group"].value_counts())

    X = sub[food_features]
    y = sub["label_prediabetes_clean2"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    pred_prob = model.predict_proba(X_test)[:,1]

    auroc = roc_auc_score(y_test, pred_prob)
    auprc = average_precision_score(y_test, pred_prob)

    print(f"\nAUROC={auroc:.3f}")
    print(f"AUPRC={auprc:.3f}")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    shap_df = pd.DataFrame({
        "feature": food_features,
        "feature_label":[label_map[x] for x in food_features],
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "age_group": age_group
    }).sort_values(
        "mean_abs_shap",
        ascending=False
    )

    print("\n[TOP SHAP]")
    print(shap_df.head(10))

    shap_df.to_csv(
        OUT_DIR / f"shap_{age_group}.csv",
        index=False
    )

    results.append({
        "age_group": age_group,
        "n": len(sub),
        "prevalence": y.mean(),
        "auroc": auroc,
        "auprc": auprc
    })

# =========================================================
# SAVE
# =========================================================

perf_df = pd.DataFrame(results)

print("\n[PERFORMANCE]")
print(perf_df)

perf_df.to_csv(
    OUT_DIR / "performance_summary.csv",
    index=False
)

print("\n[SAVED]")
print(OUT_DIR)