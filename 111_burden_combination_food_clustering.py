# ============================================================
# 111_burden_combination_food_clustering.py
#
# Food-only clustering within:
# - preDM + preHTN
# - preDM + lipid
# - preHTN + lipid
# - triple burden
#
# Outputs:
# - PCA
# - silhouette
# - clustering
# - subtype labels
# - food heatmaps
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
    "111_burden_combination_food_clustering"
)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

# ============================================================
# Food vars
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

food_vars = [v for v in food_vars if v in df.columns]

# ============================================================
# Burden combinations
# ============================================================

GROUPS = {
    "preDM_preHTN": "preDM_preHTN",
    "preDM_lipid": "preDM_lipid",
    "preHTN_lipid": "preHTN_lipid",
    "triple_burden": "triple_burden",
}

all_summary = []

for group_name, group_value in GROUPS.items():

    print("\n" + "=" * 80)
    print(group_name)

    sub = df[
        df["exclusive_predisease_group"] == group_value
    ].copy()

    print("\n[GROUP]")
    print(sub.shape)

    # --------------------------------------------------------
    # Drop NA
    # --------------------------------------------------------

    X = sub[food_vars].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    valid_idx = X.dropna().index

    sub = sub.loc[valid_idx].copy()
    X = X.loc[valid_idx].copy()

    print("\n[AFTER DROPNA]")
    print(X.shape)

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(5, X.shape[1]))

    pcs = pca.fit_transform(X_scaled)

    explained = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(pca.n_components_)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_)
    })

    print("\n[PCA EXPLAINED]")
    print(explained)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=[food_labels[v] for v in food_vars],
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    # --------------------------------------------------------
    # silhouette
    # --------------------------------------------------------

    sil_rows = []

    for k in range(2, 7):

        km = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = km.fit_predict(pcs[:, :3])

        sil = silhouette_score(
            pcs[:, :3],
            labels
        )

        sil_rows.append({
            "k": k,
            "silhouette": sil
        })

    sil_df = pd.DataFrame(sil_rows)

    best_k = sil_df.loc[
        sil_df["silhouette"].idxmax(),
        "k"
    ]

    print("\n[SILHOUETTE]")
    print(sil_df)

    print(f"\n[BEST K] {best_k}")

    # --------------------------------------------------------
    # final clustering
    # --------------------------------------------------------

    km_final = KMeans(
        n_clusters=int(best_k),
        random_state=42,
        n_init=20
    )

    clusters = km_final.fit_predict(pcs[:, :3]) + 1

    sub["food_cluster"] = clusters

    print("\n[CLUSTER COUNTS]")
    print(sub["food_cluster"].value_counts().sort_index())

    # --------------------------------------------------------
    # Food profile
    # --------------------------------------------------------

    food_profile = (
        sub.groupby("food_cluster")[food_vars]
        .mean()
    )

    food_profile.columns = [
        food_labels[c]
        for c in food_profile.columns
    ]

    print("\n[FOOD PROFILE]")
    print(food_profile.round(2))

    # --------------------------------------------------------
    # subtype labels
    # --------------------------------------------------------

    subtype_labels = {}

    for idx, row in food_profile.iterrows():

        refined_score = (
            row.get("Refined grain", 0)
            + row.get("Fast food / sweets", 0)
            + row.get("Sweet beverage", 0)
        )

        prudent_score = (
            row.get("Whole grain", 0)
            + row.get("Vegetables", 0)
            + row.get("Fruit", 0)
            + row.get("Fish / seafood", 0)
        )

        alcohol_score = (
            row.get("Alcohol", 0)
            + row.get("Coffee / tea", 0)
        )

        if prudent_score > 1:
            label = "Prudent/high-quality"

        elif alcohol_score > 1:
            label = "Alcohol/beverage"

        elif refined_score > 1:
            label = "Refined/processed"

        else:
            label = "Mixed/intermediate"

        subtype_labels[idx] = label

    sub["food_subtype"] = sub["food_cluster"].map(subtype_labels)

    print("\n[SUBTYPE LABELS]")
    print(subtype_labels)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    group_out = os.path.join(
        OUT_DIR,
        group_name
    )

    os.makedirs(group_out, exist_ok=True)

    sub.to_csv(
        os.path.join(
            group_out,
            f"111_{group_name}_food_cluster_dataset.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    explained.to_csv(
        os.path.join(
            group_out,
            f"111_{group_name}_pca_explained.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    loadings.to_csv(
        os.path.join(
            group_out,
            f"111_{group_name}_pca_loadings.csv"
        ),
        encoding="utf-8-sig"
    )

    sil_df.to_csv(
        os.path.join(
            group_out,
            f"111_{group_name}_silhouette.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    food_profile.to_csv(
        os.path.join(
            group_out,
            f"111_{group_name}_food_profile.csv"
        ),
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # PCA plot
    # --------------------------------------------------------

    plt.figure(figsize=(6, 5))

    sns.scatterplot(
        x=pcs[:, 0],
        y=pcs[:, 1],
        hue=sub["food_cluster"],
        palette="Set2",
        alpha=0.7
    )

    plt.title(
        f"Food clustering within {group_name}"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            group_out,
            f"111_{group_name}_pca_scatter.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    plt.figure(figsize=(10, 4.5))

    sns.heatmap(
        food_profile,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f"
    )

    plt.title(
        f"Food-group profile within {group_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            group_out,
            f"111_{group_name}_food_heatmap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    for k, v in subtype_labels.items():

        n_k = (
            sub[sub["food_cluster"] == k]
            .shape[0]
        )

        all_summary.append({
            "group": group_name,
            "cluster": k,
            "subtype": v,
            "n": n_k,
            "silhouette_best_k": best_k,
            "best_silhouette": sil_df["silhouette"].max()
        })

summary_df = pd.DataFrame(all_summary)

summary_df.to_csv(
    os.path.join(
        OUT_DIR,
        "111_burden_combination_cluster_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SAVED]")
print(OUT_DIR)