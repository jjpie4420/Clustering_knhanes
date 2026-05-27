# ============================================================
# 67_figure5_predm_cluster_burden_distribution.py
#
# Figure 5.
# preDM dietary subtype별 metabolic burden distribution
#
# Input:
# - 58_dataset_with_burden.csv
# - 50_predm_cluster_assigned_dataset.csv
#
# Output:
# - Figure 5A: cluster별 burden 분포
# - Figure 5B: burden별 cluster prevalence
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

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
    "67_figure5_predm_cluster_burden_distribution"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_INPUT, encoding="utf-8-sig", low_memory=False)
cluster_df = pd.read_csv(CLUSTER_INPUT, encoding="utf-8-sig", low_memory=False)

print("[DATA]")
print(df.shape)

print("[CLUSTER]")
print(cluster_df.shape)

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_df_cols = [
    "ID",
    "label_prediabetes_clean2",
    "semihealthy_burden_group",
    "semihealthy_burden_count",
]

required_cluster_cols = [
    "ID",
    "cluster",
]

missing_df = [c for c in required_df_cols if c not in df.columns]
missing_cluster = [c for c in required_cluster_cols if c not in cluster_df.columns]

if missing_df:
    raise ValueError(f"Missing columns in DATA_INPUT: {missing_df}")

if missing_cluster:
    raise ValueError(f"Missing columns in CLUSTER_INPUT: {missing_cluster}")

# ============================================================
# FILTER: preDM only
# ============================================================

df["label_prediabetes_clean2"] = pd.to_numeric(
    df["label_prediabetes_clean2"],
    errors="coerce"
)

predm = df[df["label_prediabetes_clean2"] == 1].copy()

print("\n[PREDM ONLY BEFORE MERGE]")
print(predm.shape)

# ============================================================
# MERGE CLUSTER
# ============================================================

cluster_use = cluster_df[["ID", "cluster"]].copy()

cluster_use["cluster"] = pd.to_numeric(
    cluster_use["cluster"],
    errors="coerce"
)

predm = predm.merge(
    cluster_use,
    on="ID",
    how="left"
)

predm = predm.dropna(
    subset=["cluster", "semihealthy_burden_group"]
).copy()

predm["cluster"] = predm["cluster"].astype(int)
predm["semihealthy_burden_group"] = predm["semihealthy_burden_group"].astype(str)

print("\n[PREDM AFTER MERGE]")
print(predm.shape)

print("\n[CLUSTER COUNTS]")
print(predm["cluster"].value_counts().sort_index())

print("\n[BURDEN COUNTS]")
print(predm["semihealthy_burden_group"].value_counts().sort_index())

# ============================================================
# preDM 안에서는 burden 0은 없음.
# burden 1 = preDM only
# burden 2 = preDM + one additional preclinical phenotype
# burden 3+ = preDM + preHTN + borderline lipid
# ============================================================

burden_order = ["1", "2", "3+"]
cluster_order = [1, 2, 3]

predm = predm[
    predm["semihealthy_burden_group"].isin(burden_order)
].copy()

cluster_labels = {
    1: "Cluster 1\nBeverage/refined",
    2: "Cluster 2\nBalanced/traditional",
    3: "Cluster 3\nLow-intake/intermediate",
}

# ============================================================
# TABLE 1: cluster별 burden distribution
# ============================================================

count_table = pd.crosstab(
    predm["cluster"],
    predm["semihealthy_burden_group"]
).reindex(
    index=cluster_order,
    columns=burden_order,
    fill_value=0
)

percent_by_cluster = pd.crosstab(
    predm["cluster"],
    predm["semihealthy_burden_group"],
    normalize="index"
).reindex(
    index=cluster_order,
    columns=burden_order,
    fill_value=0
) * 100

chi2, p, dof, expected = chi2_contingency(count_table)

print("\n[COUNT TABLE: CLUSTER x BURDEN]")
print(count_table)

print("\n[PERCENT BY CLUSTER]")
print(percent_by_cluster)

print("\n[CHI-SQUARE]")
print(f"chi2 = {chi2:.3f}")
print(f"p = {p:.6g}")
print(f"dof = {dof}")

# ============================================================
# FIGURE 5A
# cluster별 burden distribution
# ============================================================

plt.figure(figsize=(7.2, 4.8))

bottom = None
x = range(len(cluster_order))

for burden in burden_order:
    values = percent_by_cluster[burden].values

    plt.bar(
        x,
        values,
        bottom=bottom,
        label=f"Burden {burden}"
    )

    if bottom is None:
        bottom = values.copy()
    else:
        bottom += values

plt.xticks(
    x,
    [cluster_labels[c] for c in cluster_order]
)

plt.ylabel("Proportion within subtype (%)")
plt.xlabel("preDM dietary subtype")
plt.title(
    f"Metabolic burden distribution by preDM dietary subtype\n"
    f"Chi-square p = {p:.3g}"
)

plt.legend(
    title="Burden group",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

fig5a_path = os.path.join(
    OUT_DIR,
    "67_figure5A_cluster_burden_distribution.png"
)

plt.savefig(
    fig5a_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# TABLE 2: burden별 subtype prevalence
# ============================================================

count_table_reverse = pd.crosstab(
    predm["semihealthy_burden_group"],
    predm["cluster"]
).reindex(
    index=burden_order,
    columns=cluster_order,
    fill_value=0
)

percent_by_burden = pd.crosstab(
    predm["semihealthy_burden_group"],
    predm["cluster"],
    normalize="index"
).reindex(
    index=burden_order,
    columns=cluster_order,
    fill_value=0
) * 100

print("\n[COUNT TABLE: BURDEN x CLUSTER]")
print(count_table_reverse)

print("\n[PERCENT BY BURDEN]")
print(percent_by_burden)

# ============================================================
# FIGURE 5B
# burden별 subtype prevalence
# ============================================================

plt.figure(figsize=(7.2, 4.8))

bottom = None
x = range(len(burden_order))

for cl in cluster_order:
    values = percent_by_burden[cl].values

    plt.bar(
        x,
        values,
        bottom=bottom,
        label=cluster_labels[cl].replace("\n", " ")
    )

    if bottom is None:
        bottom = values.copy()
    else:
        bottom += values

plt.xticks(x, burden_order)

plt.ylabel("Subtype prevalence within burden group (%)")
plt.xlabel("Metabolic burden group")
plt.title("Dietary subtype prevalence across metabolic burden groups")

plt.legend(
    title="Dietary subtype",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

fig5b_path = os.path.join(
    OUT_DIR,
    "67_figure5B_burden_subtype_prevalence.png"
)

plt.savefig(
    fig5b_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# SAVE TABLES
# ============================================================

count_table.to_csv(
    os.path.join(
        OUT_DIR,
        "67_cluster_burden_count_table.csv"
    ),
    encoding="utf-8-sig"
)

percent_by_cluster.to_csv(
    os.path.join(
        OUT_DIR,
        "67_cluster_burden_percent_table.csv"
    ),
    encoding="utf-8-sig"
)

count_table_reverse.to_csv(
    os.path.join(
        OUT_DIR,
        "67_burden_cluster_count_table.csv"
    ),
    encoding="utf-8-sig"
)

percent_by_burden.to_csv(
    os.path.join(
        OUT_DIR,
        "67_burden_cluster_percent_table.csv"
    ),
    encoding="utf-8-sig"
)

# also save merged dataset
predm.to_csv(
    os.path.join(
        OUT_DIR,
        "67_predm_cluster_burden_dataset.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SAVED]")
print(OUT_DIR)

print("\n[FIGURES]")
print(fig5a_path)
print(fig5b_path)