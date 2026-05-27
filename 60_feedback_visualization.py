import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\precision_nutrition\FFQ"

BURDEN_INPUT = os.path.join(
    BASE_DIR, "results", "58_semihealthy_burden_stratification",
    "58_dataset_with_burden.csv"
)

CLUSTER_SCORE_INPUT = os.path.join(
    BASE_DIR, "results", "56_cluster_guideline_score_compare",
    "56_cluster_guideline_score_summary.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR, "results", "60_feedback_visualization"
)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(BURDEN_INPUT, encoding="utf-8-sig", low_memory=False)
cluster_score = pd.read_csv(CLUSTER_SCORE_INPUT, encoding="utf-8-sig", low_memory=False)

score_cols = [
    "score_aMED_proxy",
    "score_DASH_proxy",
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_RFS_proxy",
]

score_labels = {
    "score_aMED_proxy": "aMED",
    "score_DASH_proxy": "DASH",
    "score_AHEI_proxy": "AHEI",
    "score_HEI_proxy": "HEI",
    "score_RFS_proxy": "RFS",
}

# --------------------------------------------------
# 1. burden count distribution
# --------------------------------------------------

count_df = (
    df["semihealthy_burden_group"]
    .value_counts()
    .reindex(["0", "1", "2", "3+"])
    .reset_index()
)

count_df.columns = ["burden_group", "n"]

plt.figure(figsize=(5, 4))
plt.bar(count_df["burden_group"], count_df["n"])
plt.xlabel("Number of preclinical metabolic abnormalities")
plt.ylabel("N")
plt.title("Semihealthy burden stratification")
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "60_burden_count_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# --------------------------------------------------
# 2. guideline score by burden group
# --------------------------------------------------

for score in score_cols:
    if score not in df.columns:
        continue

    tmp = (
        df.groupby("semihealthy_burden_group", observed=False)[score]
        .agg(["mean", "std", "count"])
        .reindex(["0", "1", "2", "3+"])
        .reset_index()
    )

    tmp["se"] = tmp["std"] / np.sqrt(tmp["count"])

    plt.figure(figsize=(5, 4))
    plt.bar(
        tmp["semihealthy_burden_group"].astype(str),
        tmp["mean"],
        yerr=tmp["se"],
        capsize=4
    )
    plt.xlabel("Burden group")
    plt.ylabel(score_labels.get(score, score))
    plt.title(f"{score_labels.get(score, score)} by metabolic burden")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, f"60_{score}_by_burden.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# --------------------------------------------------
# 3. cluster guideline score profile
# --------------------------------------------------

wide = cluster_score.pivot(
    index="score",
    columns="cluster",
    values="mean"
)

wide.index = [score_labels.get(x, x) for x in wide.index]

plt.figure(figsize=(7, 4.5))

x = np.arange(len(wide.index))
width = 0.24

for i, cl in enumerate([1, 2, 3]):
    plt.bar(
        x + (i - 1) * width,
        wide[cl],
        width=width,
        label=f"Cluster {cl}"
    )

plt.xticks(x, wide.index)
plt.ylabel("Dietary score")
plt.title("Guideline-based dietary scores by preDM dietary subtype")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "60_cluster_guideline_score_barplot.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("[SAVED]")
print(OUT_DIR)