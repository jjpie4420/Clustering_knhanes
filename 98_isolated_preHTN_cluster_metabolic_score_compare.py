# ============================================================
# 98_isolated_preHTN_cluster_metabolic_score_compare.py
#
# Metabolic and dietary score comparison
# across isolated preHTN dietary subtypes
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "97_isolated_preHTN_latent_foodpattern_cluster",
    "97_isolated_preHTN_cluster_assigned_dataset.csv"
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
    "98_isolated_preHTN_cluster_metabolic_score_compare"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)
score_df = pd.read_csv(SCORE_INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

# ============================================================
# MERGE SCORES
# ============================================================

score_cols = [
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_DASH_proxy",
    "score_aMED_proxy",
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

# ============================================================
# SUBTYPE RELABEL
# ============================================================

manual_subtype_map = {
    1: "Mixed/high-intake",
    2: "Prudent/high-quality",
    3: "Beverage/alcohol",
    4: "Refined grain",
}

df["subtype_manual"] = df["cluster"].map(manual_subtype_map)

print("\n[SUBTYPE COUNTS]")
print(df["subtype_manual"].value_counts())

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

cluster_order = [1, 2, 3, 4]

# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for var in vars_to_test:

    tmp = df[["cluster", "subtype_manual", var]].dropna().copy()

    for cl in cluster_order:

        x = tmp.loc[tmp["cluster"] == cl, var]

        summary_rows.append({
            "variable": var,
            "variable_label": label_map.get(var, var),
            "cluster": cl,
            "subtype": manual_subtype_map[cl],
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "sem": x.std()/np.sqrt(len(x)),
            "median": x.median(),
            "q1": x.quantile(0.25),
            "q3": x.quantile(0.75),
        })

summary = pd.DataFrame(summary_rows)

print("\n[SUMMARY]")
print(summary)

# ============================================================
# ANOVA / KRUSKAL
# ============================================================

test_rows = []
posthoc_rows = []

for var in vars_to_test:

    tmp = df[["cluster", var]].dropna().copy()

    groups = [
        tmp.loc[tmp["cluster"] == cl, var]
        for cl in cluster_order
    ]

    groups = [g for g in groups if len(g) > 1]

    if len(groups) < 2:
        continue

    F, p_anova = f_oneway(*groups)
    H, p_kw = kruskal(*groups)

    test_rows.append({
        "variable": var,
        "variable_label": label_map.get(var, var),
        "anova_F": F,
        "anova_p": p_anova,
        "kruskal_H": H,
        "kruskal_p": p_kw,
    })

    tukey = pairwise_tukeyhsd(
        endog=tmp[var],
        groups=tmp["cluster"],
        alpha=0.05
    )

    tukey_df = pd.DataFrame(
        tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    for c in ["meandiff", "p-adj", "lower", "upper"]:
        tukey_df[c] = pd.to_numeric(
            tukey_df[c],
            errors="coerce"
        )

    tukey_df["variable"] = var
    tukey_df["variable_label"] = label_map.get(var, var)

    posthoc_rows.append(tukey_df)

tests = pd.DataFrame(test_rows)

tests["anova_fdr_bh"] = multipletests(
    tests["anova_p"],
    method="fdr_bh"
)[1]

tests["kruskal_fdr_bh"] = multipletests(
    tests["kruskal_p"],
    method="fdr_bh"
)[1]

if len(posthoc_rows) > 0:

    posthoc = pd.concat(
        posthoc_rows,
        axis=0,
        ignore_index=True
    )

    posthoc["fdr_bh"] = multipletests(
        posthoc["p-adj"],
        method="fdr_bh"
    )[1]

    posthoc["significant_fdr"] = (
        posthoc["fdr_bh"] < 0.05
    )

else:
    posthoc = pd.DataFrame()

print("\n[TESTS]")
print(tests)

print("\n[POSTHOC]")
print(posthoc.head(20))

# ============================================================
# SAVE TABLES
# ============================================================

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "98_isolated_preHTN_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

tests.to_csv(
    os.path.join(
        OUT_DIR,
        "98_isolated_preHTN_tests.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

posthoc.to_csv(
    os.path.join(
        OUT_DIR,
        "98_isolated_preHTN_posthoc.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# FIGURE 1
# METABOLIC PROFILE Z-SCORE
# ============================================================

plot_vars = [
    "HE_BMI",
    "HE_wc",
    "HE_glu",
    "HE_HbA1c",
    "HE_TG",
    "HE_HDL_st2",
    "HE_sbp",
    "HE_dbp",
]

plot_vars = [v for v in plot_vars if v in df.columns]

met_mean = (
    df.groupby("subtype_manual")[plot_vars]
    .mean()
)

met_mean.columns = [
    metabolic_labels.get(c, c)
    for c in met_mean.columns
]

met_z = met_mean.copy()

for col in met_z.columns:

    sd = met_z[col].std()

    if sd == 0 or pd.isna(sd):
        met_z[col] = 0
    else:
        met_z[col] = (
            met_z[col] - met_z[col].mean()
        ) / sd

plt.figure(figsize=(8, 5))

sns.heatmap(
    met_z,
    cmap="viridis",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Metabolic profile across isolated preHTN dietary subtypes"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "98_preHTN_metabolic_profile_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 2
# DIET SCORE
# ============================================================

score_summary = (
    df.groupby("subtype_manual")[score_cols]
    .mean()
)

score_summary.columns = [
    score_labels.get(c, c)
    for c in score_summary.columns
]

plt.figure(figsize=(7, 4.8))

sns.heatmap(
    score_summary,
    cmap="viridis",
    annot=True,
    fmt=".1f"
)

plt.title(
    "Dietary quality score across isolated preHTN subtypes"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "98_preHTN_diet_score_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n[SAVED]")
print(OUT_DIR)