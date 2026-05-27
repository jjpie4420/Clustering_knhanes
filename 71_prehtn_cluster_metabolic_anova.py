# ============================================================
# 71_prehtn_cluster_metabolic_anova.py
#
# preHTN dietary subtype별 metabolic profile 비교
# - ANOVA
# - Kruskal-Wallis
# - cluster별 summary table
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multitest import multipletests

BASE_DIR = r"D:\precision_nutrition\FFQ"

FULL_DATA_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "58_semihealthy_burden_stratification",
    "58_dataset_with_burden.csv"
)

CLUSTER_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "70_preHTN_latent_foodpattern_cluster",
    "70_preHTN_cluster_assigned_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "71_preHTN_cluster_metabolic_anova"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

full_df = pd.read_csv(FULL_DATA_INPUT, encoding="utf-8-sig", low_memory=False)
cluster_df = pd.read_csv(CLUSTER_INPUT, encoding="utf-8-sig", low_memory=False)

print("[FULL DATA]")
print(full_df.shape)

print("[CLUSTER DATA]")
print(cluster_df.shape)

# ============================================================
# MERGE
# ============================================================

if "ID" not in full_df.columns or "ID" not in cluster_df.columns:
    raise ValueError("ID column is required in both datasets.")

cluster_use = cluster_df[["ID", "cluster"]].copy()
cluster_use["cluster"] = pd.to_numeric(cluster_use["cluster"], errors="coerce")

df = full_df.merge(
    cluster_use,
    on="ID",
    how="inner"
)

print("[MERGED]")
print(df.shape)

print("\n[CLUSTER COUNTS]")
print(df["cluster"].value_counts().sort_index())

# ============================================================
# METABOLIC VARIABLES
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
    "HE_wc": "Waist circumference",
    "HE_glu": "Fasting glucose",
    "HE_HbA1c": "HbA1c",
    "HE_TG": "Triglycerides",
    "HE_chol": "Total cholesterol",
    "HE_HDL_st2": "HDL-C",
    "HE_sbp": "SBP",
    "HE_dbp": "DBP",
}

available_vars = [v for v in metabolic_vars if v in df.columns]

print("\n[AVAILABLE VARIABLES]")
print(available_vars)

for v in available_vars + ["cluster"]:
    df[v] = pd.to_numeric(df[v], errors="coerce")

df = df.dropna(subset=["cluster"]).copy()
df["cluster"] = df["cluster"].astype(int)

# ============================================================
# SUMMARY + TESTS
# ============================================================

summary_rows = []
test_rows = []

cluster_order = [1, 2, 3]

for var in available_vars:

    tmp = df[["cluster", var]].dropna().copy()

    for cl in cluster_order:
        x = tmp.loc[tmp["cluster"] == cl, var]

        summary_rows.append({
            "variable": var,
            "variable_label": metabolic_labels.get(var, var),
            "cluster": cl,
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "median": x.median(),
            "q1": x.quantile(0.25),
            "q3": x.quantile(0.75),
        })

    groups = [
        tmp.loc[tmp["cluster"] == cl, var]
        for cl in cluster_order
        if len(tmp.loc[tmp["cluster"] == cl, var]) > 0
    ]

    if len(groups) >= 2:
        f_stat, p_anova = f_oneway(*groups)
        h_stat, p_kruskal = kruskal(*groups)
    else:
        f_stat, p_anova = np.nan, np.nan
        h_stat, p_kruskal = np.nan, np.nan

    test_rows.append({
        "variable": var,
        "variable_label": metabolic_labels.get(var, var),
        "anova_F": f_stat,
        "anova_p": p_anova,
        "kruskal_H": h_stat,
        "kruskal_p": p_kruskal,
        "n_total": len(tmp),
    })

summary = pd.DataFrame(summary_rows)
tests = pd.DataFrame(test_rows)

# FDR correction
tests["anova_fdr_bh"] = multipletests(
    tests["anova_p"].fillna(1),
    method="fdr_bh"
)[1]

tests["kruskal_fdr_bh"] = multipletests(
    tests["kruskal_p"].fillna(1),
    method="fdr_bh"
)[1]

# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "71_preHTN_cluster_metabolic_profile_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

tests.to_csv(
    os.path.join(
        OUT_DIR,
        "71_preHTN_cluster_metabolic_profile_tests.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

df.to_csv(
    os.path.join(
        OUT_DIR,
        "71_preHTN_cluster_metabolic_dataset.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SUMMARY]")
print(summary)

print("\n[TESTS]")
print(tests)

print("\n[SAVED]")
print(OUT_DIR)