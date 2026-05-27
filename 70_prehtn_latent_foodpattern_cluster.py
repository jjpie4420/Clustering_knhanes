# ============================================================
# 70_prehtn_latent_foodpattern_cluster.py
#
# Dietary subtype clustering within preHTN population
#
# Figure:
# - PCA scatter
# - food-group heatmap
#
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PATH
# ============================================================

BASE_DIR = r"D:\precision_nutrition\FFQ"

DATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "58_semihealthy_burden_stratification",
    "58_dataset_with_burden.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "70_preHTN_latent_foodpattern_cluster"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)

print("\n[INPUT]")
print(df.shape)

# ============================================================
# PREHTN ONLY
# ============================================================

target_col = "label_prehypertension_clean2"

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

df = df[df[target_col] == 1].copy()

print("\n[PREHTN ONLY]")
print(df.shape)

# ============================================================
# FOOD FEATURES
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

food_labels = {
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

# ============================================================
# DROP MISSING
# ============================================================

use_cols = ["ID"] + food_features

df = df[use_cols].dropna().copy()

print("\n[AFTER DROPNA]")
print(df.shape)

# ============================================================
# X MATRIX
# ============================================================

X = df[food_features].astype(float)

# ============================================================
# PCA
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5, random_state=42)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2", "PC3", "PC4", "PC5"]
)

explained = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(5)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative": np.cumsum(pca.explained_variance_ratio_)
})

print("\n[PCA EXPLAINED]")
print(explained)

# ============================================================
# PCA LOADINGS
# ============================================================

loadings = pd.DataFrame(
    pca.components_.T,
    index=[food_labels[x] for x in food_features],
    columns=[f"PC{i+1}" for i in range(5)]
)

print("\n[PCA LOADINGS]")
print(loadings)

# ============================================================
# SILHOUETTE
# ============================================================

silhouette_results = []

for k in range(2, 7):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = km.fit_predict(X_pca[:, :2])

    sil = silhouette_score(
        X_pca[:, :2],
        labels
    )

    silhouette_results.append({
        "k": k,
        "silhouette": sil
    })

silhouette_df = pd.DataFrame(silhouette_results)

print("\n[SILHOUETTE]")
print(silhouette_df)

# ============================================================
# FINAL CLUSTER
# ============================================================

final_k = 3

kmeans = KMeans(
    n_clusters=final_k,
    random_state=42,
    n_init=20
)

cluster_labels = kmeans.fit_predict(
    X_pca[:, :2]
)

df["cluster"] = cluster_labels + 1

print("\n[CLUSTER COUNTS]")
print(df["cluster"].value_counts().sort_index())

# ============================================================
# CLUSTER PROFILE
# ============================================================

profile = (
    df.groupby("cluster")[food_features]
    .mean()
    .T
)

profile.index = [
    food_labels[x]
    for x in profile.index
]

print("\n[CLUSTER FOOD PROFILE]")
print(profile)

# ============================================================
# SAVE DATASET
# ============================================================

save_df = pd.concat(
    [
        df.reset_index(drop=True),
        pca_df.reset_index(drop=True)
    ],
    axis=1
)

save_df.to_csv(
    os.path.join(
        OUT_DIR,
        "70_preHTN_cluster_assigned_dataset.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

explained.to_csv(
    os.path.join(
        OUT_DIR,
        "70_preHTN_pca_explained.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

loadings.to_csv(
    os.path.join(
        OUT_DIR,
        "70_preHTN_pca_loadings.csv"
    ),
    encoding="utf-8-sig"
)

profile.to_csv(
    os.path.join(
        OUT_DIR,
        "70_preHTN_cluster_food_profile.csv"
    ),
    encoding="utf-8-sig"
)

# ============================================================
# FIGURE 1
# PCA SCATTER
# ============================================================

plot_df = pd.concat(
    [
        df[["cluster"]].reset_index(drop=True),
        pca_df[["PC1", "PC2"]].reset_index(drop=True)
    ],
    axis=1
)

plt.figure(figsize=(8, 6))

colors = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
}

for cl in sorted(plot_df["cluster"].unique()):

    idx = plot_df["cluster"] == cl

    plt.scatter(
        plot_df.loc[idx, "PC1"],
        plot_df.loc[idx, "PC2"],
        s=18,
        alpha=0.5,
        label=f"Cluster {cl}",
        color=colors.get(cl, None)
    )

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
)

plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
)

plt.title(
    "PreHTN dietary subtype clustering"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "70_preHTN_pca_scatter.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()
# ============================================================
# FIGURE 2
# HEATMAP
# ============================================================

heatmap_z = profile.copy()

heatmap_z = heatmap_z.apply(
    lambda x: (x - x.mean()) / x.std(),
    axis=1
)

plt.figure(figsize=(7, 6))

sns.heatmap(
    heatmap_z,
    cmap="viridis",
    center=0,
    linewidths=0.5,
    cbar_kws={
        "label": "Mean food-group z-score"
    }
)

plt.title(
    "Food-group profile by preHTN dietary subtype"
)

plt.xlabel("Cluster")
plt.ylabel("")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "70_preHTN_foodgroup_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n[SAVED]")
print(OUT_DIR)