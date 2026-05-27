# ============================================================
# 69_figure6_subtype_specific_shap.py
#
# Figure 6.
# Subtype-specific SHAP signals within preDM
#
# Question:
# 같은 preDM이라도 dietary subtype에 따라
# preHTN / borderline lipid를 설명하는 food-group signal이 다른가?
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import shap

# ============================================================
# PATH
# ============================================================

BASE_DIR = r"D:\precision_nutrition\FFQ"

DATA_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "58_semihealthy_burden_stratification",
    "58_dataset_with_burden.csv"
)

CLUSTER_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "50_predm_latent_foodpattern_cluster",
    "50_predm_cluster_assigned_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "69_figure6_subtype_specific_shap"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_INPUT, encoding="utf-8-sig", low_memory=False)
cluster_df = pd.read_csv(CLUSTER_INPUT, encoding="utf-8-sig", low_memory=False)

# ============================================================
# MERGE CLUSTER
# ============================================================

if "ID" not in df.columns or "ID" not in cluster_df.columns:
    raise ValueError("ID column is required in both datasets.")

df = df.merge(
    cluster_df[["ID", "cluster"]],
    on="ID",
    how="left"
)

# preDM only
df["label_prediabetes_clean2"] = pd.to_numeric(
    df["label_prediabetes_clean2"],
    errors="coerce"
)

df = df[df["label_prediabetes_clean2"] == 1].copy()
df = df.dropna(subset=["cluster"]).copy()
df["cluster"] = df["cluster"].astype(int)

print("[PREDM DATA]")
print(df.shape)
print(df["cluster"].value_counts().sort_index())

# ============================================================
# FEATURES / OUTCOMES
# ============================================================

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

cluster_labels = {
    1: "C1 Beverage/refined",
    2: "C2 Balanced/traditional",
    3: "C3 Low-intake/intermediate",
}

outcomes = {
    "preHTN": "label_prehypertension_clean2",
    "borderline_lipid": "label_borderline_lipid_clean",
}

# ============================================================
# MODEL + SHAP
# ============================================================

all_summary = []
all_perf = []
all_dependence = []

for outcome_name, y_col in outcomes.items():

    if y_col not in df.columns:
        print(f"[SKIP] missing outcome: {y_col}")
        continue

    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

    for cl in [1, 2, 3]:

        sub = df[df["cluster"] == cl].copy()
        sub = sub[sub[y_col].isin([0, 1])].copy()

        use_cols = food_features + [y_col]
        sub = sub[use_cols].dropna()

        print("\n" + "=" * 70)
        print(f"[OUTCOME] {outcome_name} | [CLUSTER] {cl}")
        print(sub[y_col].value_counts())

        if sub[y_col].nunique() < 2 or len(sub) < 300:
            print("[SKIP] insufficient data")
            continue

        X = sub[food_features].astype(float)
        y = sub[y_col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

        model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_train, y_train)

        pred = model.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, pred)
        auprc = average_precision_score(y_test, pred)

        all_perf.append({
            "outcome": outcome_name,
            "cluster": cl,
            "cluster_label": cluster_labels[cl],
            "n": len(sub),
            "prevalence": y.mean(),
            "auroc": auroc,
            "auprc": auprc
        })

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        shap_df = pd.DataFrame(shap_values, columns=food_features)
        x_df = X_test.reset_index(drop=True)

        for f in food_features:
            sv = shap_df[f]
            xv = x_df[f]

            high = xv >= xv.median()
            low = xv < xv.median()

            mean_high = sv[high].mean()
            mean_low = sv[low].mean()

            all_summary.append({
                "outcome": outcome_name,
                "cluster": cl,
                "cluster_label": cluster_labels[cl],
                "feature": f,
                "feature_label": feature_labels[f],
                "mean_abs_shap": np.abs(sv).mean(),
                "mean_shap": sv.mean(),
                "mean_shap_high_intake": mean_high,
                "mean_shap_low_intake": mean_low,
                "delta_high_minus_low": mean_high - mean_low,
                "direction_high_intake": "risk_positive" if mean_high > 0 else "protective_negative",
                "positive_shap_fraction": (sv > 0).mean(),
                "negative_shap_fraction": (sv < 0).mean()
            })

            # dependence data 저장
            dep_tmp = pd.DataFrame({
                "outcome": outcome_name,
                "cluster": cl,
                "feature": f,
                "x": xv.values,
                "shap": sv.values
            })

            all_dependence.append(dep_tmp)

# ============================================================
# SAVE TABLES
# ============================================================

summary = pd.DataFrame(all_summary)
perf = pd.DataFrame(all_perf)
dependence = pd.concat(all_dependence, axis=0)

summary = summary.sort_values(
    ["outcome", "cluster", "mean_abs_shap"],
    ascending=[True, True, False]
)

summary.to_csv(
    os.path.join(OUT_DIR, "69_subtype_specific_shap_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

perf.to_csv(
    os.path.join(OUT_DIR, "69_subtype_specific_model_performance.csv"),
    index=False,
    encoding="utf-8-sig"
)

dependence.to_csv(
    os.path.join(OUT_DIR, "69_subtype_specific_dependence_data.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[PERFORMANCE]")
print(perf)

print("\n[SUMMARY HEAD]")
print(summary.head(30))

# ============================================================
# FIGURE 6A
# subtype-specific SHAP top features
# outcome별 panel
# ============================================================

for outcome_name in outcomes.keys():

    tmp = summary[summary["outcome"] == outcome_name].copy()

    top_features = (
        tmp.groupby("feature_label")["mean_abs_shap"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
        .index
        .tolist()
    )

    plot_df = tmp[tmp["feature_label"].isin(top_features)].copy()

    pivot = plot_df.pivot_table(
        index="feature_label",
        columns="cluster",
        values="mean_abs_shap"
    )

    pivot = pivot.loc[top_features]

    plt.figure(figsize=(8, 5.5))

    x = np.arange(len(pivot.index))
    width = 0.24

    for i, cl in enumerate([1, 2, 3]):
        plt.bar(
            x + (i - 1) * width,
            pivot[cl],
            width=width,
            label=cluster_labels[cl]
        )

    plt.xticks(x, pivot.index, rotation=35, ha="right")
    plt.ylabel("Mean |SHAP|")
    plt.title(f"Subtype-specific SHAP importance for {outcome_name}")
    plt.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"69_figure6A_{outcome_name}_subtype_shap_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# FIGURE 6B
# direction-aware plot: delta high intake - low intake
# ============================================================

for outcome_name in outcomes.keys():

    tmp = summary[summary["outcome"] == outcome_name].copy()

    top_features = (
        tmp.groupby("feature_label")["mean_abs_shap"]
        .mean()
        .sort_values(ascending=False)
        .head(8)
        .index
        .tolist()
    )

    plot_df = tmp[tmp["feature_label"].isin(top_features)].copy()

    pivot = plot_df.pivot_table(
        index="feature_label",
        columns="cluster",
        values="delta_high_minus_low"
    )

    pivot = pivot.loc[top_features]

    plt.figure(figsize=(8, 5.5))

    x = np.arange(len(pivot.index))
    width = 0.24

    for i, cl in enumerate([1, 2, 3]):
        plt.bar(
            x + (i - 1) * width,
            pivot[cl],
            width=width,
            label=cluster_labels[cl]
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, pivot.index, rotation=35, ha="right")
    plt.ylabel("SHAP direction\n(high intake - low intake)")
    plt.title(f"Direction-aware subtype-specific SHAP for {outcome_name}")
    plt.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"69_figure6B_{outcome_name}_subtype_shap_direction.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# FIGURE 6C
# top dependence plots for selected features
# ============================================================

selected_features = [
    "fg_sweet_beverage_z",
    "fg_alcohol_z",
    "fg_kimchi_fermented_z",
    "fg_fruit_z",
]

for outcome_name in outcomes.keys():

    for f in selected_features:

        dep = dependence[
            (dependence["outcome"] == outcome_name) &
            (dependence["feature"] == f)
        ].copy()

        if dep.empty:
            continue

        plt.figure(figsize=(7, 5))

        for cl in [1, 2, 3]:
            subdep = dep[dep["cluster"] == cl]

            plt.scatter(
                subdep["x"],
                subdep["shap"],
                s=14,
                alpha=0.45,
                label=cluster_labels[cl]
            )

        plt.axhline(0, linestyle="--", linewidth=1)
        plt.xlabel(feature_labels[f])
        plt.ylabel("SHAP value")
        plt.title(f"{feature_labels[f]} dependence for {outcome_name}")
        plt.legend(fontsize=8)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUT_DIR,
                f"69_figure6C_{outcome_name}_{f}_dependence.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

print("\n[SAVED]")
print(OUT_DIR)