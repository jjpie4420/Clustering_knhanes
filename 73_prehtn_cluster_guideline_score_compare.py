# ============================================================
# 73_prehtn_cluster_guideline_score_compare.py
#
# preHTN dietary subtype별 guideline-based dietary score 비교
# - aMED, DASH, AHEI, HEI, RFS
# - ANOVA / Kruskal
# - Tukey posthoc
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

BASE_DIR = r"D:\precision_nutrition\FFQ"

SCORE_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "55_guideline_diet_scores",
    "55_guideline_diet_scores_dataset.csv"
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
    "73_preHTN_cluster_guideline_score_compare"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

score_df = pd.read_csv(SCORE_INPUT, encoding="utf-8-sig", low_memory=False)
cluster_df = pd.read_csv(CLUSTER_INPUT, encoding="utf-8-sig", low_memory=False)

print("[SCORE DATA]")
print(score_df.shape)

print("[CLUSTER DATA]")
print(cluster_df.shape)

# ============================================================
# SCORE COLUMNS
# ============================================================

score_cols = [
    "score_aMED_proxy",
    "score_DASH_proxy",
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_RFS_proxy",
]

score_cols = [c for c in score_cols if c in score_df.columns]

print("\n[SCORE COLS]")
print(score_cols)

# ============================================================
# MERGE
# ============================================================

if "ID" not in score_df.columns or "ID" not in cluster_df.columns:
    raise ValueError("ID column is required in both datasets.")

df = cluster_df[["ID", "cluster"]].merge(
    score_df[["ID"] + score_cols],
    on="ID",
    how="left"
)

for c in ["cluster"] + score_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["cluster"]).copy()
df["cluster"] = df["cluster"].astype(int)

print("\n[MERGED]")
print(df.shape)

print("\n[CLUSTER COUNTS]")
print(df["cluster"].value_counts().sort_index())

# ============================================================
# SUMMARY / TESTS / POSTHOC
# ============================================================

cluster_order = [1, 2, 3]

summary_rows = []
test_rows = []
posthoc_rows = []

for score in score_cols:

    tmp = df[["cluster", score]].dropna().copy()

    for cl in cluster_order:
        x = tmp.loc[tmp["cluster"] == cl, score]

        summary_rows.append({
            "score": score,
            "cluster": cl,
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "median": x.median(),
            "q1": x.quantile(0.25),
            "q3": x.quantile(0.75),
        })

    groups = [
        tmp.loc[tmp["cluster"] == cl, score]
        for cl in cluster_order
        if len(tmp.loc[tmp["cluster"] == cl, score]) > 0
    ]

    if len(groups) >= 2:
        f_stat, p_anova = f_oneway(*groups)
        h_stat, p_kruskal = kruskal(*groups)
    else:
        f_stat, p_anova = np.nan, np.nan
        h_stat, p_kruskal = np.nan, np.nan

    test_rows.append({
        "score": score,
        "anova_F": f_stat,
        "anova_p": p_anova,
        "kruskal_H": h_stat,
        "kruskal_p": p_kruskal,
        "n_total": len(tmp),
    })

    if tmp["cluster"].nunique() >= 2:
        tukey = pairwise_tukeyhsd(
            endog=tmp[score],
            groups=tmp["cluster"].astype(str),
            alpha=0.05
        )

        tukey_df = pd.DataFrame(
            tukey.summary().data[1:],
            columns=tukey.summary().data[0]
        )

        tukey_df["score"] = score
        posthoc_rows.append(tukey_df)

summary = pd.DataFrame(summary_rows)
tests = pd.DataFrame(test_rows)
posthoc = pd.concat(posthoc_rows, axis=0)

# numeric conversion
for c in ["meandiff", "p-adj", "lower", "upper"]:
    if c in posthoc.columns:
        posthoc[c] = pd.to_numeric(posthoc[c], errors="coerce")

# FDR correction
tests["anova_fdr_bh"] = multipletests(
    tests["anova_p"].fillna(1),
    method="fdr_bh"
)[1]

tests["kruskal_fdr_bh"] = multipletests(
    tests["kruskal_p"].fillna(1),
    method="fdr_bh"
)[1]

posthoc["fdr_bh"] = multipletests(
    posthoc["p-adj"].fillna(1),
    method="fdr_bh"
)[1]

posthoc["significant_fdr"] = posthoc["fdr_bh"] < 0.05

# ============================================================
# SAVE
# ============================================================

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "73_preHTN_cluster_guideline_score_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

tests.to_csv(
    os.path.join(
        OUT_DIR,
        "73_preHTN_cluster_guideline_score_tests.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

posthoc.to_csv(
    os.path.join(
        OUT_DIR,
        "73_preHTN_cluster_guideline_score_posthoc.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

df.to_csv(
    os.path.join(
        OUT_DIR,
        "73_preHTN_cluster_guideline_score_dataset.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SUMMARY]")
print(summary)

print("\n[TESTS]")
print(tests)

print("\n[POSTHOC]")
print(posthoc)

print("\n[SAVED]")
print(OUT_DIR)