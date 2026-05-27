import os
import numpy as np
import pandas as pd

import shap
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

INPUT = r"D:\precision_nutrition\FFQ\processed\analysis_cohort_semihealthy_foodgroups.csv"
OUT_DIR = r"D:\precision_nutrition\FFQ\results\40_borderline_lipid_foodgroup_shap"

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

print("[GROUP]")
print(sub["lipid_group"].value_counts())

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

X = sub[food_features].copy()
y = sub["target"].astype(int)

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

model.fit(X_imp, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_imp)

mean_abs_shap = np.abs(shap_values).mean(axis=0)
mean_shap = shap_values.mean(axis=0)

summary = pd.DataFrame({
    "feature": food_features,
    "feature_label": [feature_labels[f] for f in food_features],
    "mean_abs_shap": mean_abs_shap,
    "mean_shap": mean_shap,
    "direction": np.where(mean_shap > 0, "positive", "negative"),
}).sort_values("mean_abs_shap", ascending=False)

summary.to_csv(
    f"{OUT_DIR}/40_borderline_lipid_shap_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[SHAP SUMMARY]")
print(summary)

plt.figure()
shap.summary_plot(
    shap_values,
    X_imp,
    feature_names=[feature_labels[f] for f in food_features],
    show=False
)

plt.tight_layout()
plt.savefig(
    f"{OUT_DIR}/40_borderline_lipid_shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

top_features = summary["feature"].head(5).tolist()

for feat in top_features:
    plt.figure()
    shap.dependence_plot(
        feat,
        shap_values,
        X_imp,
        interaction_index=None,
        feature_names=food_features,
        show=False
    )
    plt.xlabel(feature_labels.get(feat, feat))
    plt.tight_layout()
    plt.savefig(
        f"{OUT_DIR}/{feat}_dependence.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

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
    q_summary["feature_label"] = feature_labels.get(feat, feat)

    rows.append(q_summary)

quantile_summary = pd.concat(rows, axis=0)

quantile_summary.to_csv(
    f"{OUT_DIR}/40_borderline_lipid_quantile_shap_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[QUANTILE SHAP SUMMARY]")
print(quantile_summary)

print("\n[SAVED]")
print(OUT_DIR)