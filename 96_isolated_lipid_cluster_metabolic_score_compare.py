# ============================================================
# 96_isolated_lipid_cluster_metabolic_score_compare.py
#
# Compare metabolic profile and dietary scores between
# isolated lipid dietary subtypes:
# - Cluster 1: Prudent/high-quality
# - Cluster 2: Refined/beverage
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

BASE_DIR = r"D:\precision_nutrition\FFQ"

DATA_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "95_isolated_lipid_latent_foodpattern_cluster",
    "95_isolated_lipid_cluster_assigned_dataset.csv"
)

SCORE_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "55_guideline_diet_scores",
    "55_guideline_diet_scores_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "96_isolated_lipid_cluster_metabolic_score_compare"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(DATA_INPUT, encoding="utf-8-sig", low_memory=False)
score_df = pd.read_csv(SCORE_INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

# merge dietary scores
score_cols = [
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_DASH_proxy",
    "score_aMED_proxy",
    # RFS는 현재 proxy 문제로 제외
    # "score_RFS_proxy",
]

score_cols = [c for c in score_cols if c in score_df.columns]

df = df.merge(
    score_df[["ID"] + score_cols],
    on="ID",
    how="left",
    suffixes=("", "_score")
)

print("\n[MERGED]")
print(df.shape)

print("\n[CLUSTER COUNTS]")
print(df["cluster"].value_counts().sort_index())

print("\n[SUBTYPE COUNTS]")
print(df["subtype"].value_counts())

# ============================================================
# VARIABLES
# ============================================================

metabolic_vars = [
    "age",
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

metabolic_labels = {
    "age": "Age",
    "HE_BMI": "BMI",
    "HE_wc": "Waist",
    "HE_glu": "Glucose",
    "HE_HbA1c": "HbA1c",
    "HE_TG": "TG",
    "HE_chol": "Total cholesterol",
    "HE_HDL_st2": "HDL-C",
    "HE_sbp": "SBP",
    "HE_dbp": "DBP",
}

score_labels = {
    "score_AHEI_proxy": "AHEI",
    "score_HEI_proxy": "HEI",
    "score_DASH_proxy": "DASH",
    "score_aMED_proxy": "aMED",
}

vars_to_test = [v for v in metabolic_vars if v in df.columns] + score_cols

label_map = {**metabolic_labels, **score_labels}

for v in vars_to_test:
    df[v] = pd.to_numeric(df[v], errors="coerce")

# ============================================================
# SUMMARY + TEST
# ============================================================

summary_rows = []
test_rows = []

cluster_order = [1, 2]

for var in vars_to_test:

    tmp = df[["cluster", "subtype", var]].dropna().copy()

    for cl in cluster_order:
        x = tmp.loc[tmp["cluster"] == cl, var]

        subtype_name = (
            tmp.loc[tmp["cluster"] == cl, "subtype"].iloc[0]
            if len(tmp.loc[tmp["cluster"] == cl]) > 0
            else ""
        )

        summary_rows.append({
            "variable": var,
            "variable_label": label_map.get(var, var),
            "cluster": cl,
            "subtype": subtype_name,
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "sem": x.std() / np.sqrt(len(x)) if len(x) > 0 else np.nan,
            "median": x.median(),
            "q1": x.quantile(0.25),
            "q3": x.quantile(0.75),
        })

    x1 = tmp.loc[tmp["cluster"] == 1, var]
    x2 = tmp.loc[tmp["cluster"] == 2, var]

    if len(x1) > 1 and len(x2) > 1:
        t_stat, t_p = ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
        u_stat, u_p = mannwhitneyu(x1, x2, alternative="two-sided")
    else:
        t_stat, t_p, u_stat, u_p = np.nan, np.nan, np.nan, np.nan

    diff_2_minus_1 = x2.mean() - x1.mean()

    test_rows.append({
        "variable": var,
        "variable_label": label_map.get(var, var),
        "cluster1_subtype": "Prudent/high-quality",
        "cluster2_subtype": "Refined/beverage",
        "mean_cluster1": x1.mean(),
        "mean_cluster2": x2.mean(),
        "diff_cluster2_minus_cluster1": diff_2_minus_1,
        "ttest_stat": t_stat,
        "ttest_p": t_p,
        "mannwhitney_u": u_stat,
        "mannwhitney_p": u_p,
        "n_cluster1": len(x1),
        "n_cluster2": len(x2),
    })

summary = pd.DataFrame(summary_rows)
tests = pd.DataFrame(test_rows)

tests["ttest_fdr_bh"] = multipletests(
    tests["ttest_p"].fillna(1),
    method="fdr_bh"
)[1]

tests["mannwhitney_fdr_bh"] = multipletests(
    tests["mannwhitney_p"].fillna(1),
    method="fdr_bh"
)[1]

tests["significant_ttest_fdr"] = tests["ttest_fdr_bh"] < 0.05
tests["significant_mannwhitney_fdr"] = tests["mannwhitney_fdr_bh"] < 0.05

# ============================================================
# SAVE TABLES
# ============================================================

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "96_isolated_lipid_subtype_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

tests.to_csv(
    os.path.join(
        OUT_DIR,
        "96_isolated_lipid_subtype_tests.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SUMMARY]")
print(summary)

print("\n[TESTS]")
print(tests)

# ============================================================
# FIGURE 1: metabolic profile relative z-score
# ============================================================

plot_met_vars = [
    "HE_BMI",
    "HE_wc",
    "HE_glu",
    "HE_HbA1c",
    "HE_TG",
    "HE_HDL_st2",
    "HE_sbp",
    "HE_dbp",
]

plot_met_vars = [v for v in plot_met_vars if v in df.columns]

met_mean = (
    df.groupby("cluster")[plot_met_vars]
    .mean()
    .T
)

met_mean.index = [metabolic_labels.get(v, v) for v in met_mean.index]

met_z = met_mean.copy()

for idx in met_z.index:
    row = met_z.loc[idx]
    sd = row.std()
    if sd == 0 or pd.isna(sd):
        met_z.loc[idx] = 0
    else:
        met_z.loc[idx] = (row - row.mean()) / sd

plt.figure(figsize=(7.5, 4.8))

x = np.arange(len(met_z.index))
width = 0.34

plt.bar(
    x - width / 2,
    met_z[1],
    width=width,
    label="Prudent/high-quality"
)

plt.bar(
    x + width / 2,
    met_z[2],
    width=width,
    label="Refined/beverage"
)

plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(x, met_z.index, rotation=35, ha="right")
plt.ylabel("Relative level between subtypes")
plt.title("Metabolic profile in isolated lipid dietary subtypes")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "96_isolated_lipid_metabolic_profile_zscore.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 2: dietary score barplot
# ============================================================

score_plot_cols = score_cols

if score_plot_cols:

    score_summary = (
        df.groupby("cluster")[score_plot_cols]
        .agg(["mean", "sem"])
    )

    score_means = df.groupby("cluster")[score_plot_cols].mean()
    score_sems = df.groupby("cluster")[score_plot_cols].sem()

    plt.figure(figsize=(7.2, 4.8))

    x = np.arange(len(score_plot_cols))
    width = 0.34

    plt.bar(
        x - width / 2,
        score_means.loc[1, score_plot_cols],
        yerr=score_sems.loc[1, score_plot_cols],
        width=width,
        capsize=4,
        label="Prudent/high-quality"
    )

    plt.bar(
        x + width / 2,
        score_means.loc[2, score_plot_cols],
        yerr=score_sems.loc[2, score_plot_cols],
        width=width,
        capsize=4,
        label="Refined/beverage"
    )

    plt.xticks(
        x,
        [score_labels.get(c, c) for c in score_plot_cols]
    )

    plt.ylabel("Dietary quality score")
    plt.title("Dietary quality scores in isolated lipid dietary subtypes")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            "96_isolated_lipid_diet_score_barplot.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# FIGURE 3: key marker barplots with p-values
# ============================================================

key_vars = [
    "HE_TG",
    "HE_HDL_st2",
    "HE_wc",
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_DASH_proxy",
    "score_aMED_proxy",
]

key_vars = [v for v in key_vars if v in vars_to_test]

for var in key_vars:

    tmp = summary[summary["variable"] == var].copy()
    test_row = tests[tests["variable"] == var].iloc[0]

    plt.figure(figsize=(4.8, 4.2))

    x_labels = [
        "Prudent/\nhigh-quality",
        "Refined/\nbeverage"
    ]

    plt.bar(
        [0, 1],
        tmp.sort_values("cluster")["mean"],
        yerr=tmp.sort_values("cluster")["sem"],
        capsize=4
    )

    p = test_row["mannwhitney_fdr_bh"]

    plt.xticks([0, 1], x_labels)
    plt.ylabel(label_map.get(var, var))
    plt.title(f"{label_map.get(var, var)}\nFDR-adjusted p={p:.3g}")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"96_isolated_lipid_{var}_barplot.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

print("\n[SAVED]")
print(OUT_DIR)