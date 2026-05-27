# ============================================================
# Figure 5.
# Burden distribution across dietary subtypes
# - cluster별 metabolic burden 분포
# - burden별 subtype prevalence
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

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

df = pd.read_csv(DATA_INPUT, encoding="utf-8-sig", low_memory=False)
cluster_df = pd.read_csv(CLUSTER_INPUT, encoding="utf-8-sig", low_memory=False)

print("[DATA]")
print(df.shape)
print("[CLUSTER]")
print(cluster_df.shape)

# preDM only
df = df[df["label_prediabetes_clean2"] == 1].copy()

# cluster merge
if "ID" in df.columns and "ID" in cluster_df.columns:
    df = df.merge(
        cluster_df[["ID", "cluster"]],
        on="ID",
        how="left"
    )
else:
    print("[WARNING] ID 없음. row-order fallback 사용.")
    df = df.reset_index(drop=True)
    cluster_df = cluster_df.reset_index(drop=True)
    df["cluster"] = cluster_df["cluster"]

df = df.dropna(subset=["cluster", "semihealthy_burden_group"]).copy()
df["cluster"] = df["cluster"].astype(int)
df["semihealthy_burden_group"] = df["semihealthy_burden_group"].astype(str)

print("[MERGED PRE-DM]")
print(df.shape)
print(df["cluster"].value_counts().sort_index())
print(df["semihealthy_burden_group"].value_counts().sort_index())

# ============================================================
# BURDEN COUNT
# ============================================================

# category
def burden_group(x):
    if x == 0:
        return "0"
    elif x == 1:
        return "1"
    elif x == 2:
        return "2"
    else:
        return "3"

df["burden_group"] = df["metabolic_burden_n"].apply(burden_group)

print("\n[BURDEN DISTRIBUTION]")
print(df["burden_group"].value_counts())

# ============================================================
# PRE-DM ONLY
# ============================================================

predm = df[df["label_prediabetes_clean"] == 1].copy()

print("\n[PRE-DM]")
print(predm.shape)

# ============================================================
# LOAD CLUSTER RESULT
# ============================================================

cluster_df = pd.read_csv(
    r"D:\precision_nutrition\FFQ\results\50_predm_dietary_subtype\predm_cluster_labels.csv"
)

print("\n[CLUSTER]")
print(cluster_df.shape)

# merge
predm = predm.merge(
    cluster_df[["ID", "cluster"]],
    on="ID",
    how="left"
)

print("\n[MERGED]")
print(predm.shape)

# ============================================================
# FIGURE 5A
# cluster별 burden distribution
# ============================================================

ct = pd.crosstab(
    predm["cluster"],
    predm["burden_group"],
    normalize="index"
) * 100

print("\n[CROSSTAB]")
print(ct)

# ============================================================
# CHI-SQUARE
# ============================================================

chi2_table = pd.crosstab(
    predm["cluster"],
    predm["burden_group"]
)

chi2, p, dof, expected = chi2_contingency(chi2_table)

print("\n[CHI-SQUARE]")
print(f"chi2={chi2:.3f}")
print(f"p={p:.6g}")

# ============================================================
# PLOT 1
# stacked burden distribution by cluster
# ============================================================

fig, ax = plt.subplots(figsize=(8,6))

ct.plot(
    kind="bar",
    stacked=True,
    ax=ax
)

ax.set_ylabel("Percent (%)")
ax.set_xlabel("Dietary subtype cluster")
ax.set_title(
    f"Metabolic burden distribution by preDM dietary subtype\n"
    f"Chi-square p={p:.3g}"
)

plt.legend(
    title="Burden group",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

save_path = (
    r"D:\precision_nutrition\FFQ\results"
    r"\65_burden_distribution_cluster"
)

import os
os.makedirs(save_path, exist_ok=True)

plt.savefig(
    os.path.join(
        save_path,
        "figure5_cluster_burden_distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 5B
# burden별 subtype prevalence
# ============================================================

ct2 = pd.crosstab(
    predm["burden_group"],
    predm["cluster"],
    normalize="index"
) * 100

print("\n[REVERSE TABLE]")
print(ct2)

fig, ax = plt.subplots(figsize=(8,6))

ct2.plot(
    kind="bar",
    stacked=True,
    ax=ax
)

ax.set_ylabel("Percent (%)")
ax.set_xlabel("Metabolic burden group")
ax.set_title(
    "Dietary subtype prevalence across metabolic burden groups"
)

plt.legend(
    title="Cluster",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        save_path,
        "figure5_burden_subtype_prevalence.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# EXPORT TABLE
# ============================================================

ct.to_csv(
    os.path.join(
        save_path,
        "cluster_burden_distribution.csv"
    )
)

ct2.to_csv(
    os.path.join(
        save_path,
        "burden_subtype_prevalence.csv"
    )
)

print("\n[SAVED]")
print(save_path)