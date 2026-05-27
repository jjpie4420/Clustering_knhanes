import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import shap

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR, "results", "55_guideline_diet_scores",
    "55_guideline_diet_scores_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR, "results", "59_shap_direction_by_phenotype_sex"
)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

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
    "fg_coffee_tea_z":"Coffee / tea",
}

phenotypes = {
    "preHTN": "label_prehypertension_clean2",
    "preDM": "label_prediabetes_clean2",
    "borderline_lipid": "label_borderline_lipid_clean",
}

all_summary = []
all_perf = []

for pheno, y_col in phenotypes.items():

    if y_col not in df.columns:
        print(f"[SKIP] {pheno}: {y_col} not found")
        continue

    for sex_value, sex_name in [(1, "Male"), (2, "Female"), ("all", "All")]:

        sub = df.copy()

        if sex_value != "all" and "sex" in sub.columns:
            sub = sub[sub["sex"] == sex_value].copy()

        sub = sub[sub[y_col].isin([0, 1])].copy()

        use_cols = food_features + [y_col]
        sub = sub[use_cols].dropna()

        if sub[y_col].nunique() < 2 or len(sub) < 300:
            print(f"[SKIP] {pheno} {sex_name}: insufficient data")
            continue

        X = sub[food_features].astype(float)
        y = sub[y_col].astype(int)

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

        pred = model.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, pred)
        auprc = average_precision_score(y_test, pred)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        shap_df = pd.DataFrame(shap_values, columns=food_features)
        x_df = X_test.reset_index(drop=True)

        rows = []

        for f in food_features:
            sv = shap_df[f]
            xv = x_df[f]

            # high intake group 기준 방향성
            high = xv >= xv.median()
            low = xv < xv.median()

            mean_shap_high = sv[high].mean()
            mean_shap_low = sv[low].mean()

            direction_high_intake = (
                "risk_positive" if mean_shap_high > 0 else "protective_negative"
            )

            rows.append({
                "phenotype": pheno,
                "sex": sex_name,
                "feature": f,
                "feature_label": label_map.get(f, f),
                "mean_abs_shap": np.abs(sv).mean(),
                "mean_shap": sv.mean(),
                "mean_shap_high_intake": mean_shap_high,
                "mean_shap_low_intake": mean_shap_low,
                "delta_high_minus_low": mean_shap_high - mean_shap_low,
                "direction_high_intake": direction_high_intake,
                "positive_shap_fraction": (sv > 0).mean(),
                "negative_shap_fraction": (sv < 0).mean(),
            })

        out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)

        all_summary.append(out)

        all_perf.append({
            "phenotype": pheno,
            "sex": sex_name,
            "n": len(sub),
            "prevalence": y.mean(),
            "auroc": auroc,
            "auprc": auprc
        })

        print(f"\n[{pheno} / {sex_name}]")
        print(f"AUROC={auroc:.3f}, AUPRC={auprc:.3f}")
        print(out.head(10))

summary = pd.concat(all_summary, axis=0)
perf = pd.DataFrame(all_perf)

summary.to_csv(
    os.path.join(OUT_DIR, "59_shap_direction_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

perf.to_csv(
    os.path.join(OUT_DIR, "59_model_performance.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SAVED]")
print(OUT_DIR)