import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INPUT = r"D:\precision_nutrition\FFQ\processed\analysis_cohort_semihealthy_foodgroups.csv"
OUT_DIR = r"D:\precision_nutrition\FFQ\results\50_predm_latent_foodpattern_cluster"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

# --------------------------------------------------
# preDM only
# --------------------------------------------------

sub = df[df["glucose_group"] == "prediabetes_clean"].copy()

print("[PREDM ONLY]")
print(sub.shape)

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

clinical_vars = [
    "age",
    "sex",
    "HE_BMI",
    "HE_wc",
    "HE_glu",
    "HE_HbA1c",
    "HE_TG",
    "HE_chol",
    "HE_HDL_st2",
    "HE_sbp",
    "HE_dbp",
]

clinical_vars = [c for c in clinical_vars if c in sub.columns]

for c in food_features + clinical_vars:
    sub[c] = pd.to_numeric(sub[c], errors="coerce")

# --------------------------------------------------
# X matrix
# --------------------------------------------------

X = sub[food_features].copy()

imp = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_imp = imp.fit_transform(X)
X_scaled = scaler.fit_transform(X_imp)

# --------------------------------------------------
# PCA
# --------------------------------------------------

pca = PCA(n_components=5, random_state=42)
pcs = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    pcs,
    columns=[f"PC{i+1}" for i in range(pcs.shape[1])]
)

pca_df["ID"] = sub["ID"].values if "ID" in sub.columns else np.arange(len(sub))

explained = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative": np.cumsum(pca.explained_variance_ratio_)
})

loadings = pd.DataFrame(
    pca.components_.T,
    index=[feature_labels[f] for f in food_features],
    columns=[f"PC{i+1}" for i in range(pcs.shape[1])]
)

explained.to_csv(
    f"{OUT_DIR}/50_pca_explained_variance.csv",
    index=False,
    encoding="utf-8-sig"
)

loadings.to_csv(
    f"{OUT_DIR}/50_pca_loadings.csv",
    encoding="utf-8-sig"
)

print("\n[PCA EXPLAINED]")
print(explained)

print("\n[PCA LOADINGS]")
print(loadings)

# --------------------------------------------------
# choose K by silhouette
# --------------------------------------------------

sil_rows = []

for k in range(2, 7):
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=50
    )

    labels = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels)

    sil_rows.append({
        "k": k,
        "silhouette": sil
    })

sil_df = pd.DataFrame(sil_rows)

sil_df.to_csv(
    f"{OUT_DIR}/50_kmeans_silhouette.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[SILHOUETTE]")
print(sil_df)

# --------------------------------------------------
# final K
# --------------------------------------------------

# 기본값: 3개 subtype
FINAL_K = 3

km = KMeans(
    n_clusters=FINAL_K,
    random_state=42,
    n_init=100
)

sub["cluster"] = km.fit_predict(X_scaled)

# cluster label을 1부터 시작
sub["cluster"] = sub["cluster"] + 1

print("\n[CLUSTER COUNTS]")
print(sub["cluster"].value_counts().sort_index())

# --------------------------------------------------
# cluster food profile
# --------------------------------------------------

cluster_food = (
    sub.groupby("cluster")[food_features]
    .mean()
    .T
)

cluster_food.index = [feature_labels[f] for f in food_features]

cluster_food.to_csv(
    f"{OUT_DIR}/50_cluster_foodgroup_profile.csv",
    encoding="utf-8-sig"
)

print("\n[CLUSTER FOOD PROFILE]")
print(cluster_food)

# --------------------------------------------------
# clinical profile by cluster
# --------------------------------------------------

clinical_summary = []

for cl in sorted(sub["cluster"].dropna().unique()):
    tmp = sub[sub["cluster"] == cl]

    row = {
        "cluster": cl,
        "n": len(tmp),
        "percent": len(tmp) / len(sub) * 100,
    }

    for c in clinical_vars:
        row[f"{c}_mean"] = tmp[c].mean()
        row[f"{c}_sd"] = tmp[c].std()

    if "sex" in tmp.columns:
        row["female_percent"] = (tmp["sex"] == 2).mean() * 100

    clinical_summary.append(row)

clinical_summary = pd.DataFrame(clinical_summary)

clinical_summary.to_csv(
    f"{OUT_DIR}/50_cluster_clinical_profile.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[CLUSTER CLINICAL PROFILE]")
print(clinical_summary)

# --------------------------------------------------
# PCA scatter
# --------------------------------------------------

plot_df = pca_df.copy()
plot_df["cluster"] = sub["cluster"].values

plt.figure(figsize=(7, 6))

for cl in sorted(plot_df["cluster"].unique()):
    tmp = plot_df[plot_df["cluster"] == cl]
    plt.scatter(
        tmp["PC1"],
        tmp["PC2"],
        s=12,
        alpha=0.5,
        label=f"Cluster {cl}"
    )

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Prediabetes dietary subtype clustering")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/50_pca_cluster_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# heatmap-like profile plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(
    cluster_food.values,
    aspect="auto"
)

plt.yticks(
    ticks=np.arange(len(cluster_food.index)),
    labels=cluster_food.index
)

plt.xticks(
    ticks=np.arange(cluster_food.shape[1]),
    labels=[f"Cluster {c}" for c in cluster_food.columns]
)

plt.colorbar(label="Mean food-group z-score")
plt.title("Food-group profile by preDM dietary subtype")
plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/50_cluster_foodgroup_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# save cluster-assigned dataset
# --------------------------------------------------

keep_cols = (
    ["ID", "cluster", "glucose_group"]
    if "ID" in sub.columns
    else ["cluster", "glucose_group"]
)

keep_cols += food_features + clinical_vars

sub[keep_cols].to_csv(
    f"{OUT_DIR}/50_predm_cluster_assigned_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[SAVED]")
print(OUT_DIR)