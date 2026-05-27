# ============================================================
# 108_integrated_preDM_cluster_characterization.py
#
# Characterization of integrated preDM phenotypes
# Purpose:
# - demographic / dietary quality / metabolic profiling
# - NOT metabolic validation
# ============================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "105_isolated_preDM_integrated_diet_metabolic_cluster",
    "105_isolated_preDM_integrated_cluster_assigned_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "108_integrated_preDM_cluster_characterization"
)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

cluster_col = "integrated_cluster"

print("\n[CLUSTER COUNTS]")
print(df[cluster_col].value_counts().sort_index())

# ============================================================
# MANUAL LABELS
# 105 결과 기준
# ============================================================

manual_label_map = {
    1: "Lifestyle-metabolic burden phenotype",
    2: "Refined dietary phenotype",
    3: "Prudent-metabolically resilient phenotype",
}

df["integrated_label"] = df[cluster_col].map(manual_label_map)

# ============================================================
# VARIABLES
# ============================================================

candidate_vars = {
    "Age": ["age"],
    "BMI": ["HE_BMI"],
    "Waist": ["HE_wc"],

    "Glucose": ["HE_glu"],
    "HbA1c": ["HE_HbA1c"],
    "TG": ["HE_TG"],
    "HDL-C": ["HE_HDL_st2"],
    "SBP": ["HE_sbp"],
    "DBP": ["HE_dbp"],

    "AHEI": ["score_AHEI_proxy"],
    "HEI": ["score_HEI_proxy"],
    "DASH": ["score_DASH_proxy"],
    "aMED": ["score_aMED_proxy"],
    "RFS": ["score_RFS_proxy"],

    "Sleep duration": ["sleep_hour", "sleep_duration"],
    "Exercise": ["exercise_time", "PA_MET"],
}

resolved = {}

for label, candidates in candidate_vars.items():
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break

    if found is not None:
        resolved[label] = found
    else:
        print(f"[MISSING] {label}")

print("\n[RESOLVED VARIABLES]")
print(resolved)

# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for label, col in resolved.items():

    tmp = df[[cluster_col, "integrated_label", col]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.dropna()

    for cl in sorted(tmp[cluster_col].dropna().unique()):

        vals = tmp.loc[tmp[cluster_col] == cl, col]

        summary_rows.append({
            "variable": label,
            "cluster": cl,
            "cluster_label": manual_label_map.get(int(cl), f"cluster {cl}"),
            "n": len(vals),
            "mean": vals.mean(),
            "sd": vals.std(),
            "median": vals.median(),
            "q1": vals.quantile(0.25),
            "q3": vals.quantile(0.75),
        })

summary_df = pd.DataFrame(summary_rows)

print("\n[SUMMARY]")
print(summary_df.head(20))

# ============================================================
# ANOVA / KRUSKAL
# ============================================================

test_rows = []

for label, col in resolved.items():

    tmp = df[[cluster_col, col]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.dropna()

    groups = [
        tmp.loc[tmp[cluster_col] == cl, col].values
        for cl in sorted(tmp[cluster_col].dropna().unique())
    ]

    groups = [g for g in groups if len(g) > 1]

    if len(groups) < 2:
        continue

    try:
        F, p_anova = f_oneway(*groups)
    except Exception:
        F, p_anova = np.nan, np.nan

    try:
        H, p_kw = kruskal(*groups)
    except Exception:
        H, p_kw = np.nan, np.nan

    test_rows.append({
        "variable": label,
        "anova_F": F,
        "anova_p": p_anova,
        "kruskal_H": H,
        "kruskal_p": p_kw,
    })

test_df = pd.DataFrame(test_rows)

if not test_df.empty:
    test_df["anova_fdr_bh"] = multipletests(
        test_df["anova_p"].fillna(1),
        method="fdr_bh"
    )[1]

    test_df["kruskal_fdr_bh"] = multipletests(
        test_df["kruskal_p"].fillna(1),
        method="fdr_bh"
    )[1]

print("\n[TEST RESULTS]")
print(test_df)

# ============================================================
# POSTHOC
# ============================================================

posthoc_rows = []

for label, col in resolved.items():

    tmp = df[[cluster_col, col]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.dropna()

    if tmp[cluster_col].nunique() < 2:
        continue

    try:
        tukey = pairwise_tukeyhsd(
            endog=tmp[col],
            groups=tmp[cluster_col],
            alpha=0.05
        )

        tukey_df = pd.DataFrame(
            tukey.summary().data[1:],
            columns=tukey.summary().data[0]
        )

        tukey_df["variable"] = label
        posthoc_rows.append(tukey_df)

    except Exception:
        pass

if len(posthoc_rows) > 0:
    posthoc_df = pd.concat(posthoc_rows, axis=0, ignore_index=True)

    if "p-adj" in posthoc_df.columns:
        posthoc_df["p-adj"] = pd.to_numeric(posthoc_df["p-adj"], errors="coerce")
        posthoc_df["fdr_bh"] = multipletests(
            posthoc_df["p-adj"].fillna(1),
            method="fdr_bh"
        )[1]
        posthoc_df["significant_fdr"] = posthoc_df["fdr_bh"] < 0.05
else:
    posthoc_df = pd.DataFrame()

print("\n[POSTHOC]")
print(posthoc_df.head(30))

# ============================================================
# HEATMAP
# ============================================================

heatmap_df = summary_df.pivot(
    index="variable",
    columns="cluster_label",
    values="mean"
)

col_order = [
    "Lifestyle-metabolic burden phenotype",
    "Refined dietary phenotype",
    "Prudent-metabolically resilient phenotype",
]
col_order = [c for c in col_order if c in heatmap_df.columns]
heatmap_df = heatmap_df[col_order]

heatmap_z = heatmap_df.apply(
    lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x * 0,
    axis=1
)

plt.figure(figsize=(9.5, 8.5))

sns.heatmap(
    heatmap_z,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Integrated preDM phenotype characterization\n"
    "(z-score standardized mean)"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "108_integrated_preDM_characterization_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# BOXPLOTS
# ============================================================

plot_vars = [
    "Glucose",
    "HbA1c",
    "TG",
    "HDL-C",
    "BMI",
    "AHEI",
    "HEI",
    "DASH",
]

for v in plot_vars:

    if v not in resolved:
        continue

    col = resolved[v]

    tmp = df[[cluster_col, "integrated_label", col]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.dropna()

    plt.figure(figsize=(8, 4.8))

    sns.boxplot(
        data=tmp,
        x="integrated_label",
        y=col
    )

    plt.xticks(rotation=25, ha="right")
    plt.title(v)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"108_boxplot_{v}.png"
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
        "108_integrated_preDM_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    os.path.join(
        OUT_DIR,
        "108_integrated_preDM_tests.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

posthoc_df.to_csv(
    os.path.join(
        OUT_DIR,
        "108_integrated_preDM_posthoc.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

heatmap_df.to_csv(
    os.path.join(
        OUT_DIR,
        "108_integrated_preDM_heatmap_raw.csv"
    ),
    encoding="utf-8-sig"
)

heatmap_z.to_csv(
    os.path.join(
        OUT_DIR,
        "108_integrated_preDM_heatmap_zscore.csv"
    ),
    encoding="utf-8-sig"
)

print("\n[SAVED]")
print(OUT_DIR)