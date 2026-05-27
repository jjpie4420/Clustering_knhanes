# ============================================================
# 106_integrated_lipid_cluster_characterization.py
#
# Characterization of integrated lipid phenotypes
#
# Purpose:
# - demographic / lifestyle / dietary-quality profiling
# - NOT metabolic validation
#
# Important:
# metabolic variables were already used in clustering
# ============================================================

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import f_oneway
from scipy.stats import kruskal

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "103_isolated_lipid_integrated_diet_metabolic_cluster",
    "103_isolated_lipid_integrated_cluster_assigned_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "106_integrated_lipid_cluster_characterization"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

cluster_col = "integrated_cluster"

print("\n[CLUSTER COUNTS]")
print(df[cluster_col].value_counts().sort_index())

# ============================================================
# OPTIONAL MANUAL LABEL
# ============================================================

manual_label_map = {
    1: "Prudent-lower metabolic risk",
    2: "Refined-metabolic risk",
}

df["integrated_label"] = (
    df[cluster_col]
    .map(manual_label_map)
)

# ============================================================
# VARIABLES FOR CHARACTERIZATION
# ============================================================

candidate_vars = {

    # demographic
    "Age": ["age", "AGE", "Age"],
    "BMI": ["HE_BMI"],
    "Waist": ["HE_wc", "waist_circumference"],

    # glucose/lipid profile
    "Glucose": ["HE_glu"],
    "HbA1c": ["HE_HbA1c"],
    "TG": ["HE_TG"],
    "HDL-C": ["HE_HDL_st2"],

    # blood pressure
    "SBP": ["HE_sbp"],
    "DBP": ["HE_dbp"],

    # dietary quality
    "AHEI": ["score_AHEI_proxy"],
    "HEI": ["score_HEI_proxy"],
    "DASH": ["score_DASH_proxy"],
    "aMED": ["score_aMED_proxy"],
    "RFS": ["score_RFS_proxy"],

    # lifestyle
    "Sleep duration": ["sleep_hour", "sleep_duration"],
    "Exercise": ["exercise_time", "PA_MET"],

}

# ============================================================
# VARIABLE RESOLUTION
# ============================================================

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

    tmp = df[[cluster_col, col]].copy()

    tmp[col] = pd.to_numeric(
        tmp[col],
        errors="coerce"
    )

    tmp = tmp.dropna()

    for cl in sorted(tmp[cluster_col].unique()):

        sub = tmp[
            tmp[cluster_col] == cl
        ][col]

        summary_rows.append({
            "variable": label,
            "cluster": cl,
            "cluster_label": manual_label_map.get(cl),
            "n": len(sub),
            "mean": sub.mean(),
            "sd": sub.std(),
            "median": sub.median(),
            "q1": sub.quantile(0.25),
            "q3": sub.quantile(0.75)
        })

summary_df = pd.DataFrame(summary_rows)

print("\n[SUMMARY]")
print(summary_df.head())

# ============================================================
# ANOVA / KRUSKAL
# ============================================================

test_rows = []

for label, col in resolved.items():

    tmp = df[[cluster_col, col]].copy()

    tmp[col] = pd.to_numeric(
        tmp[col],
        errors="coerce"
    )

    tmp = tmp.dropna()

    groups = []

    for cl in sorted(tmp[cluster_col].unique()):

        vals = tmp[
            tmp[cluster_col] == cl
        ][col].values

        groups.append(vals)

    if len(groups) < 2:
        continue

    try:
        F, p_anova = f_oneway(*groups)
    except:
        F, p_anova = np.nan, np.nan

    try:
        H, p_kw = kruskal(*groups)
    except:
        H, p_kw = np.nan, np.nan

    test_rows.append({
        "variable": label,
        "anova_F": F,
        "anova_p": p_anova,
        "kruskal_H": H,
        "kruskal_p": p_kw
    })

test_df = pd.DataFrame(test_rows)

if not test_df.empty:

    test_df["anova_fdr_bh"] = multipletests(
        test_df["anova_p"],
        method="fdr_bh"
    )[1]

    test_df["kruskal_fdr_bh"] = multipletests(
        test_df["kruskal_p"],
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

    tmp[col] = pd.to_numeric(
        tmp[col],
        errors="coerce"
    )

    tmp = tmp.dropna()

    n_cluster = (
        tmp[cluster_col]
        .nunique()
    )

    if n_cluster < 2:
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

    except:
        pass

if len(posthoc_rows) > 0:

    posthoc_df = pd.concat(
        posthoc_rows,
        axis=0,
        ignore_index=True
    )

    if "p-adj" in posthoc_df.columns:

        posthoc_df["fdr_bh"] = multipletests(
            pd.to_numeric(
                posthoc_df["p-adj"],
                errors="coerce"
            ),
            method="fdr_bh"
        )[1]

else:

    posthoc_df = pd.DataFrame()

print("\n[POSTHOC]")
print(posthoc_df.head())

# ============================================================
# HEATMAP PROFILE
# ============================================================

heatmap_df = (
    summary_df
    .pivot(
        index="variable",
        columns="cluster_label",
        values="mean"
    )
)

heatmap_z = (
    heatmap_df
    .apply(
        lambda x: (
            x - x.mean()
        ) / x.std(),
        axis=1
    )
)

plt.figure(figsize=(7.5, 8.5))

sns.heatmap(
    heatmap_z,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title(
    "Integrated lipid phenotype characterization\n(z-score standardized mean)"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "106_integrated_lipid_characterization_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# BOXPLOT
# ============================================================

plot_vars = [
    "TG",
    "HDL-C",
    "BMI",
    "AHEI",
    "HEI",
]

for v in plot_vars:

    if v not in resolved:
        continue

    col = resolved[v]

    tmp = df[[cluster_col, "integrated_label", col]].copy()

    tmp[col] = pd.to_numeric(
        tmp[col],
        errors="coerce"
    )

    tmp = tmp.dropna()

    plt.figure(figsize=(5.5, 4.5))

    sns.boxplot(
        data=tmp,
        x="integrated_label",
        y=col
    )

    plt.xticks(rotation=10)

    plt.title(v)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"106_boxplot_{v}.png"
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
        "106_integrated_lipid_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    os.path.join(
        OUT_DIR,
        "106_integrated_lipid_tests.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

posthoc_df.to_csv(
    os.path.join(
        OUT_DIR,
        "106_integrated_lipid_posthoc.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

heatmap_df.to_csv(
    os.path.join(
        OUT_DIR,
        "106_integrated_lipid_heatmap_raw.csv"
    ),
    encoding="utf-8-sig"
)

heatmap_z.to_csv(
    os.path.join(
        OUT_DIR,
        "106_integrated_lipid_heatmap_zscore.csv"
    ),
    encoding="utf-8-sig"
)

print("\n[SAVED]")
print(OUT_DIR)