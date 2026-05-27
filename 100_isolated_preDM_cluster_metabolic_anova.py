# ============================================================
# 100_isolated_preDM_cluster_metabolic_anova.py
#
# Metabolic comparison across isolated preDM dietary subtypes
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "99_isolated_preDM_latent_foodpattern_cluster",
    "99_isolated_preDM_cluster_assigned_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "100_isolated_preDM_cluster_metabolic_anova"
)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

# ============================================================
# ONLY CLUSTERED SUBJECTS
# ============================================================

df = df[df["cluster"].notna()].copy()

print("\n[CLUSTERED]")
print(df.shape)

print("\n[CLUSTER COUNTS]")
print(df["cluster"].value_counts().sort_index())

# ============================================================
# METABOLIC VARIABLES
# ============================================================

meta_vars = {
    "HE_BMI": "BMI",
    "waist_circumference": "Waist",
    "fasting_glucose": "Glucose",
    "HbA1c": "HbA1c",
    "TG": "TG",
    "HDL_C": "HDL-C",
    "SBP": "SBP",
    "DBP": "DBP",
    "age": "Age"
}

# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for var, label in meta_vars.items():

    if var not in df.columns:
        print(f"[MISSING] {var}")
        continue

    tmp = df[["cluster", var]].copy()

    tmp[var] = pd.to_numeric(tmp[var], errors="coerce")

    tmp = tmp.dropna()

    for cl in sorted(tmp["cluster"].unique()):

        vals = tmp.loc[tmp["cluster"] == cl, var]

        summary_rows.append({
            "variable": label,
            "cluster": cl,
            "n": len(vals),
            "mean": vals.mean(),
            "sd": vals.std(),
            "median": vals.median(),
            "q1": vals.quantile(0.25),
            "q3": vals.quantile(0.75)
        })

summary_df = pd.DataFrame(summary_rows)

print("\n[SUMMARY]")
print(summary_df)

# ============================================================
# ANOVA + KRUSKAL
# ============================================================

test_rows = []

for var, label in meta_vars.items():

    if var not in df.columns:
        continue

    tmp = df[["cluster", var]].copy()

    tmp[var] = pd.to_numeric(tmp[var], errors="coerce")

    tmp = tmp.dropna()

    groups = []

    for cl in sorted(tmp["cluster"].unique()):

        vals = tmp.loc[tmp["cluster"] == cl, var]

        groups.append(vals)

    if len(groups) < 2:
        continue

    # ANOVA
    F, p_anova = f_oneway(*groups)

    # Kruskal
    H, p_kw = kruskal(*groups)

    test_rows.append({
        "variable": label,
        "anova_F": F,
        "anova_p": p_anova,
        "kruskal_H": H,
        "kruskal_p": p_kw,
        "n_total": len(tmp)
    })

tests_df = pd.DataFrame(test_rows)

# FDR
tests_df["anova_fdr_bh"] = multipletests(
    tests_df["anova_p"],
    method="fdr_bh"
)[1]

tests_df["kruskal_fdr_bh"] = multipletests(
    tests_df["kruskal_p"],
    method="fdr_bh"
)[1]

print("\n[TESTS]")
print(tests_df)

# ============================================================
# POSTHOC
# ============================================================

posthoc_rows = []

sig_vars = tests_df.loc[
    tests_df["anova_fdr_bh"] < 0.05,
    "variable"
].tolist()

reverse_map = {
    v: k for k, v in meta_vars.items()
}

for label in sig_vars:

    var = reverse_map[label]

    tmp = df[["cluster", var]].copy()

    tmp[var] = pd.to_numeric(tmp[var], errors="coerce")

    tmp = tmp.dropna()

    tukey = pairwise_tukeyhsd(
        endog=tmp[var],
        groups=tmp["cluster"],
        alpha=0.05
    )

    tukey_df = pd.DataFrame(
        data=tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    tukey_df["variable"] = label

    posthoc_rows.append(tukey_df)

if len(posthoc_rows) > 0:

    posthoc_df = pd.concat(
        posthoc_rows,
        axis=0,
        ignore_index=True
    )

    posthoc_df["p-adj"] = pd.to_numeric(
        posthoc_df["p-adj"],
        errors="coerce"
    )

    posthoc_df["fdr_bh"] = multipletests(
        posthoc_df["p-adj"],
        method="fdr_bh"
    )[1]

    posthoc_df["significant_fdr"] = (
        posthoc_df["fdr_bh"] < 0.05
    )

else:

    posthoc_df = pd.DataFrame()

print("\n[POSTHOC]")
print(posthoc_df)

# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(
    os.path.join(
        OUT_DIR,
        "100_metabolic_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

tests_df.to_csv(
    os.path.join(
        OUT_DIR,
        "100_metabolic_tests.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

posthoc_df.to_csv(
    os.path.join(
        OUT_DIR,
        "100_metabolic_posthoc.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SAVED]")
print(OUT_DIR)