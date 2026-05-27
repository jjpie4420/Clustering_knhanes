# ============================================================
# 68_figure6_same_burden_subtype_analysis.py
#
# Figure 6.
# Dietary heterogeneity within same metabolic burden level
#
# Target:
# - burden 2 subgroup only
# - PCA + clustering
# - heatmap
# - dietary score comparison
#
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

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ============================================================
# PATH
# ============================================================

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "58_semihealthy_burden_stratification",
    "58_dataset_with_burden.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "68_figure6_same_burden_subtype"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

# ============================================================
# FILTER
# burden 2 only
# preDM only
# ============================================================

df = df[
    (df["label_prediabetes_clean2"] == 1) &
    (df["semihealthy_burden_group"] == "2")
].copy()

print("\n[BURDEN 2 ONLY]")
print(df.shape)

# ============================================================
# FOOD GROUPS
# ============================================================

food_cols = [
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
# PCA
# ============================================================

X = df[food_cols].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_

print("\n[PCA EXPLAINED]")
print(explained)

# ============================================================
# K selection
# ============================================================

silhouette_results = []

for k in range(2, 6):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = km.fit_predict(X_pca[:, :3])

    sil = silhouette_score(
        X_pca[:, :3],
        labels
    )

    silhouette_results.append([k, sil])

sil_df = pd.DataFrame(
    silhouette_results,
    columns=["k", "silhouette"]
)

print("\n[SILHOUETTE]")
print(sil_df)

# ============================================================
# FINAL CLUSTER
# ============================================================

FINAL_K = 3

kmeans = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=20
)

df["cluster"] = kmeans.fit_predict(
    X_pca[:, :3]
) + 1

print("\n[CLUSTER COUNTS]")
print(df["cluster"].value_counts())

# ============================================================
# FIGURE 6A
# PCA scatter
# ============================================================

plt.figure(figsize=(6.5, 5.5))

for c in sorted(df["cluster"].unique()):

    idx = df["cluster"] == c

    plt.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        s=18,
        alpha=0.6,
        label=f"Cluster {c}"
    )

plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")

plt.title(
    "Dietary subtype clustering within burden 2 preDM subgroup"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "68_figure6A_burden2_pca_scatter.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 6B
# Heatmap
# ============================================================

heatmap_df = (
    df.groupby("cluster")[food_cols]
    .mean()
    .T
)

heatmap_df.index = [
    food_labels[x]
    for x in heatmap_df.index
]

plt.figure(figsize=(6.5, 7))

sns.heatmap(
    heatmap_df,
    cmap="viridis",
    center=0,
    annot=False
)

plt.title(
    "Food-group profile within burden 2 preDM subgroup"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "68_figure6B_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# DIETARY SCORES
# ============================================================

score_cols = [
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_DASH_proxy",
    "score_aMED_proxy",
]

summary_results = []
anova_results = []
posthoc_results = []

for score in score_cols:

    tmp = df[["cluster", score]].dropna()

    # summary
    s = (
        tmp.groupby("cluster")[score]
        .agg(["count", "mean", "std"])
        .reset_index()
    )

    s["score"] = score

    summary_results.append(s)

    # anova
    groups = [
        tmp[tmp["cluster"] == c][score]
        for c in sorted(tmp["cluster"].unique())
    ]

    F, p = f_oneway(*groups)

    anova_results.append([
        score,
        F,
        p
    ])

    # tukey
    tukey = pairwise_tukeyhsd(
        endog=tmp[score],
        groups=tmp["cluster"],
        alpha=0.05
    )

    tukey_df = pd.DataFrame(
        tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    tukey_df["score"] = score

    posthoc_results.append(tukey_df)

summary_df = pd.concat(summary_results)
anova_df = pd.DataFrame(
    anova_results,
    columns=["score", "F", "p"]
)

posthoc_df = pd.concat(posthoc_results)

print("\n[ANOVA]")
print(anova_df)

# ============================================================
# FIGURE 6C
# dietary score barplot
# ============================================================

plot_scores = [
    ("score_AHEI_proxy", "AHEI"),
    ("score_HEI_proxy", "HEI"),
    ("score_DASH_proxy", "DASH"),
    ("score_aMED_proxy", "aMED"),
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(9, 7)
)

axes = axes.flatten()

for ax, (score, label) in zip(axes, plot_scores):

    plot_df = (
        df.groupby("cluster")[score]
        .agg(["mean", "sem"])
        .reset_index()
    )

    ax.bar(
        plot_df["cluster"].astype(str),
        plot_df["mean"],
        yerr=plot_df["sem"],
        capsize=4
    )

    ax.set_title(label)
    ax.set_xlabel("Cluster")

plt.suptitle(
    "Guideline-based dietary scores within burden 2 preDM subgroup",
    y=1.02,
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "68_figure6C_dietscore_barplot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(
    os.path.join(
        OUT_DIR,
        "68_dietscore_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

anova_df.to_csv(
    os.path.join(
        OUT_DIR,
        "68_dietscore_anova.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

posthoc_df.to_csv(
    os.path.join(
        OUT_DIR,
        "68_dietscore_posthoc.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

df.to_csv(
    os.path.join(
        OUT_DIR,
        "68_burden2_cluster_dataset.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SAVED]")
print(OUT_DIR)