# ============================================================
# 99_isolated_preDM_latent_foodpattern_cluster.py
#
# PCA + KMeans clustering within
# mutually exclusive isolated preDM group
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

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "99_isolated_preDM_latent_foodpattern_cluster"
)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

# ============================================================
# ISOLATED preDM ONLY
# ============================================================

df = df[
    df["exclusive_predisease_group"] == "isolated_preDM"
].copy()

print("\n[ISOLATED preDM ONLY]")
print(df.shape)

# ============================================================
# FOOD VARIABLES
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

tmp = df[["ID"] + food_vars].copy()

for c in food_vars:
    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

tmp = tmp.dropna()

print("\n[AFTER DROPNA]")
print(tmp.shape)

# ============================================================
# STANDARDIZE + PCA
# ============================================================

X = tmp[food_vars].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=[f"PC{i+1}" for i in range(5)],
    index=tmp.index
)

explained = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(5)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative": np.cumsum(pca.explained_variance_ratio_)
})

loadings = pd.DataFrame(
    pca.components_.T,
    index=[food_labels[c] for c in food_vars],
    columns=[f"PC{i+1}" for i in range(5)]
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
        n_init=20
    )

    cluster = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, cluster)

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
# FINAL CLUSTER
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

cluster = kmeans.fit_predict(X_pca)

tmp["cluster"] = cluster + 1
pca_df["cluster"] = cluster + 1

print("\n[CLUSTER COUNTS]")
print(tmp["cluster"].value_counts().sort_index())

# ============================================================
# FOOD PROFILE
# ============================================================

food_profile = (
    tmp.groupby("cluster")[food_vars]
    .mean()
)

food_profile.columns = [
    food_labels[c]
    for c in food_profile.columns
]

print("\n[FOOD PROFILE]")
print(food_profile)

# ============================================================
# AUTO SUBTYPE LABEL
# ============================================================

subtype_labels = {}

for cl in food_profile.index:

    row = food_profile.loc[cl]

    refined = (
        row["Refined grain"]
        + row["Sweet beverage"]
        + row["Fast food / sweets"]
        + row["Coffee / tea"] * 0.5
    )

    prudent = (
        row["Whole grain"]
        + row["Vegetables"]
        + row["Fruit"]
        + row["Fish / seafood"]
        + row["Dairy"] * 0.5
    )

    high_intake = row.mean()

    if refined > prudent + 0.5:
        label = "Refined/beverage"
    elif prudent > refined + 0.5:
        label = "Prudent/high-quality"
    elif high_intake > 0.2:
        label = "Mixed/high-intake"
    else:
        label = "Low-intake/intermediate"

    subtype_labels[cl] = label

tmp["subtype"] = tmp["cluster"].map(subtype_labels)

print("\n[SUBTYPE LABELS]")
print(subtype_labels)

# ============================================================
# SAVE
# ============================================================

save_df = df.merge(
    tmp[["ID", "cluster", "subtype"]],
    on="ID",
    how="left"
)

save_df.to_csv(
    os.path.join(
        OUT_DIR,
        "99_isolated_preDM_cluster_assigned_dataset.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

food_profile.to_csv(
    os.path.join(
        OUT_DIR,
        "99_isolated_preDM_food_profile.csv"
    ),
    encoding="utf-8-sig"
)

sil_df.to_csv(
    os.path.join(
        OUT_DIR,
        "99_isolated_preDM_silhouette.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

explained.to_csv(
    os.path.join(
        OUT_DIR,
        "99_isolated_preDM_pca_explained.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

loadings.to_csv(
    os.path.join(
        OUT_DIR,
        "99_isolated_preDM_pca_loadings.csv"
    ),
    encoding="utf-8-sig"
)

# ============================================================
# FIGURE 1 PCA
# ============================================================

plt.figure(figsize=(6.5, 5.5))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="Set2",
    alpha=0.7,
    s=24
)

plt.title("Dietary subtype clustering within isolated preDM group")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "99_pca_cluster_plot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 2 FOOD PROFILE HEATMAP
# ============================================================

plt.figure(figsize=(6.8, 4.8))

sns.heatmap(
    food_profile,
    cmap="viridis",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title("Food-group profile by isolated preDM subtype")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "99_food_profile_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n[SAVED]")
print(OUT_DIR)