import os
import numpy as np
import pandas as pd
import shap

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

INPUT = r"D:\precision_nutrition\FFQ\processed\analysis_cohort_semihealthy_foodgroups.csv"
OUT_DIR = r"D:\precision_nutrition\FFQ\results\42_borderline_lipid_sex_stratified"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

sub = df[df["lipid_group"].isin([
    "normal_lipid",
    "borderline_lipid_clean"
])].copy()

sub["target"] = np.where(
    sub["lipid_group"] == "borderline_lipid_clean",
    1,
    0
)

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
    "fg_coffee_tea_z",
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
    "fg_coffee_tea_z": "Coffee / tea",
}

results = []

for sex_value, sex_label in [(1, "Male"), (2, "Female")]:
    print("\n==============================")
    print(f"[SEX] {sex_label}")
    print("==============================")

    tmp = sub[sub["sex"] == sex_value].copy()

    print(tmp["lipid_group"].value_counts())

    X = tmp[food_features].copy()
    y = tmp["target"].astype(int)

    for c in food_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imp.fit_transform(X),
        columns=food_features
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

    model.fit(X_imp, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_imp)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "sex": sex_label,
        "feature": food_features,
        "feature_label": [feature_labels[f] for f in food_features],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    shap_df.to_csv(
        f"{OUT_DIR}/{sex_label}_borderline_lipid_shap_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[TOP SHAP]")
    print(shap_df.head(10))

    results.append({
        "sex": sex_label,
        "n": len(tmp),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "prevalence": y.mean(),
        "auroc": auroc,
        "auprc": auprc,
    })

perf = pd.DataFrame(results)

perf.to_csv(
    f"{OUT_DIR}/42_borderline_lipid_sex_stratified_performance.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[PERFORMANCE]")
print(perf)

print("\n[SAVED]")
print(OUT_DIR)