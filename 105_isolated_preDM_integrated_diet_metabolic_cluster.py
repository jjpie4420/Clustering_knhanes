# ============================================================
# 105_isolated_preDM_integrated_diet_metabolic_cluster.py
#
# Diet-metabolic integrated clustering within isolated preDM
#
# Features:
# - food-group variables
# - metabolic variables: BMI, glucose, HbA1c, TG, HDL-C, SBP, DBP
#
# Mild exclusion:
# - remove overt diabetes-range outliers
#   HE_glu >= 126 or HE_HbA1c >= 6.5
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "94_mutually_exclusive_predisease_groups",
    "94_dataset_with_exclusive_predisease_groups.csv"
)

FOOD_ONLY_CLUSTER = os.path.join(
    BASE_DIR,
    "results",
    "99_isolated_preDM_latent_foodpattern_cluster",
    "99_isolated_preDM_cluster_assigned_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "105_isolated_preDM_integrated_diet_metabolic_cluster"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

df = df[
    df["exclusive_predisease_group"] == "isolated_preDM"
].copy()

print("\n[ISOLATED preDM ONLY]")
print(df.shape)

# ============================================================
# MILD HYPERGLYCEMIC OUTLIER EXCLUSION
# ============================================================

df["HE_glu"] = pd.to_numeric(df["HE_glu"], errors="coerce")
df["HE_HbA1c"] = pd.to_numeric(df["HE_HbA1c"], errors="coerce")

df = df[
    (df["HE_glu"] < 126) &
    (df["HE_HbA1c"] < 6.5)
].copy()

print("\n[AFTER MILD HYPERGLYCEMIC EXCLUSION]")
print(df.shape)

# ============================================================
# FEATURE SET
# ============================================================

food_vars = [
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

metabolic_vars = [
    "HE_BMI",
    "HE_glu",
    "HE_HbA1c",
    "HE_TG",
    "HE_HDL_st2",
    "HE_sbp",
    "HE_dbp",
]

feature_vars = food_vars + metabolic_vars

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
    "HE_BMI": "BMI",
    "HE_glu": "Glucose",
    "HE_HbA1c": "HbA1c",
    "HE_TG": "TG",
    "HE_HDL_st2": "HDL-C",
    "HE_sbp": "SBP",
    "HE_dbp": "DBP",
}

missing = [c for c in feature_vars if c not in df.columns]
if missing:
    raise ValueError(f"Missing variables: {missing}")

# ============================================================
# CLEAN
# ============================================================

tmp = df[["ID"] + feature_vars].copy()

for c in feature_vars:
    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

tmp = tmp.dropna()

print("\n[AFTER DROPNA]")
print(tmp.shape)

# ============================================================
# STANDARDIZE
# ============================================================

X = tmp[feature_vars].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# PCA
# ============================================================

pca = PCA(n_components=8, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]

pca_df = pd.DataFrame(
    X_pca,
    columns=pca_cols,
    index=tmp.index
)

explained = pd.DataFrame({
    "PC": pca_cols,
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative": np.cumsum(pca.explained_variance_ratio_)
})

loadings = pd.DataFrame(
    pca.components_.T,
    index=[feature_labels[c] for c in feature_vars],
    columns=pca_cols
)

print("\n[PCA EXPLAINED]")
print(explained)

print("\n[PCA LOADINGS]")
print(loadings)

# ============================================================
# SILHOUETTE
# ============================================================

sil_rows = []

for k in range(2, 7):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=30
    )

    labels = km.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)

    sil_rows.append({
        "k": k,
        "silhouette": sil
    })

sil_df = pd.DataFrame(sil_rows)

print("\n[SILHOUETTE]")
print(sil_df)

best_k = int(
    sil_df.sort_values(
        "silhouette",
        ascending=False
    ).iloc[0]["k"]
)

print(f"\n[BEST K] {best_k}")

# ============================================================
# FINAL KMEANS
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=30
)

cluster = kmeans.fit_predict(X_pca)

tmp["integrated_cluster"] = cluster + 1
pca_df["integrated_cluster"] = cluster + 1

print("\n[INTEGRATED CLUSTER COUNTS]")
print(tmp["integrated_cluster"].value_counts().sort_index())

# ============================================================
# PROFILE
# ============================================================

profile = (
    tmp.groupby("integrated_cluster")[feature_vars]
    .mean()
)

profile.columns = [
    feature_labels[c]
    for c in profile.columns
]

food_profile = profile[
    [feature_labels[c] for c in food_vars]
].copy()

metabolic_profile = profile[
    [feature_labels[c] for c in metabolic_vars]
].copy()

print("\n[INTEGRATED FEATURE PROFILE]")
print(profile)

print("\n[FOOD PROFILE]")
print(food_profile)

print("\n[METABOLIC PROFILE]")
print(metabolic_profile)

# ============================================================
# PLACEHOLDER LABELS
# 결과 확인 후 manual label 권장
# ============================================================

phenotype_labels = {}

for cl in profile.index:
    phenotype_labels[cl] = f"Integrated phenotype {cl}"

tmp["integrated_phenotype"] = (
    tmp["integrated_cluster"]
    .map(phenotype_labels)
)

print("\n[PLACEHOLDER PHENOTYPE LABELS]")
print(phenotype_labels)

# ============================================================
# MERGE FOOD-ONLY CLUSTER
# ============================================================

if os.path.exists(FOOD_ONLY_CLUSTER):

    food_only = pd.read_csv(
        FOOD_ONLY_CLUSTER,
        encoding="utf-8-sig",
        low_memory=False
    )

    food_only_cols = ["ID", "cluster", "subtype"]
    food_only_cols = [
        c for c in food_only_cols
        if c in food_only.columns
    ]

    food_only = food_only[food_only_cols].copy()

    food_only = food_only.rename(columns={
        "cluster": "food_only_cluster",
        "subtype": "food_only_subtype"
    })

    tmp = tmp.merge(
        food_only,
        on="ID",
        how="left"
    )

    print("\n[FOOD-ONLY VS INTEGRATED CROSSTAB]")
    print(pd.crosstab(
        tmp["food_only_subtype"],
        tmp["integrated_cluster"]
    ))

# ============================================================
# SAVE
# ============================================================

save_df = df.merge(
    tmp[[
        "ID",
        "integrated_cluster",
        "integrated_phenotype"
    ] + [
        c for c in [
            "food_only_cluster",
            "food_only_subtype"
        ]
        if c in tmp.columns
    ]],
    on="ID",
    how="left"
)

save_df.to_csv(
    os.path.join(
        OUT_DIR,
        "105_isolated_preDM_integrated_cluster_assigned_dataset.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

profile.to_csv(
    os.path.join(
        OUT_DIR,
        "105_integrated_feature_profile.csv"
    ),
    encoding="utf-8-sig"
)

food_profile.to_csv(
    os.path.join(
        OUT_DIR,
        "105_integrated_food_profile.csv"
    ),
    encoding="utf-8-sig"
)

metabolic_profile.to_csv(
    os.path.join(
        OUT_DIR,
        "105_integrated_metabolic_profile.csv"
    ),
    encoding="utf-8-sig"
)

sil_df.to_csv(
    os.path.join(
        OUT_DIR,
        "105_integrated_silhouette.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

explained.to_csv(
    os.path.join(
        OUT_DIR,
        "105_integrated_pca_explained.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

loadings.to_csv(
    os.path.join(
        OUT_DIR,
        "105_integrated_pca_loadings.csv"
    ),
    encoding="utf-8-sig"
)

# ============================================================
# FIGURE 1 PCA SCATTER
# ============================================================

plt.figure(figsize=(6.8, 5.6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="integrated_cluster",
    palette="Set2",
    alpha=0.70,
    s=24
)

plt.title(
    "Integrated diet-metabolic phenotype clustering\n"
    "within isolated preDM group"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "105_integrated_pca_cluster_plot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 2 FOOD PROFILE
# ============================================================

plt.figure(figsize=(7.8, 4.8))

sns.heatmap(
    food_profile,
    cmap="viridis",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Food profile by integrated preDM phenotype"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "105_integrated_food_profile_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 3 METABOLIC PROFILE
# ============================================================

plt.figure(figsize=(6.5, 4.6))

sns.heatmap(
    metabolic_profile,
    cmap="viridis",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Metabolic profile by integrated preDM phenotype"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "105_integrated_metabolic_profile_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 4 COMBINED PROFILE
# ============================================================

plt.figure(figsize=(11.5, 5.5))

sns.heatmap(
    profile,
    cmap="viridis",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Integrated diet-metabolic profile within isolated preDM group"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "105_integrated_combined_profile_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n[SAVED]")
print(OUT_DIR)