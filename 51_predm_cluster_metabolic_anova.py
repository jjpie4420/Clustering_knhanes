import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal

INPUT = r"D:\precision_nutrition\FFQ\results\50_predm_latent_foodpattern_cluster\50_predm_cluster_assigned_dataset.csv"
OUT_DIR = r"D:\precision_nutrition\FFQ\results\51_predm_cluster_metabolic_anova"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)
print(df["cluster"].value_counts().sort_index())

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

metabolic_vars = [c for c in metabolic_vars if c in df.columns]

for c in metabolic_vars + ["cluster"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

summary_rows = []
test_rows = []

for var in metabolic_vars:
    tmp = df[["cluster", var]].dropna()

    cluster_values = [
        tmp.loc[tmp["cluster"] == cl, var]
        for cl in sorted(tmp["cluster"].dropna().unique())
    ]

    # descriptive
    for cl in sorted(tmp["cluster"].dropna().unique()):
        x = tmp.loc[tmp["cluster"] == cl, var]

        summary_rows.append({
            "variable": var,
            "cluster": int(cl),
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "median": x.median(),
            "q1": x.quantile(0.25),
            "q3": x.quantile(0.75),
        })

    # one-way ANOVA
    try:
        f_stat, p_anova = f_oneway(*cluster_values)
    except Exception:
        f_stat, p_anova = np.nan, np.nan

    # Kruskal-Wallis
    try:
        h_stat, p_kruskal = kruskal(*cluster_values)
    except Exception:
        h_stat, p_kruskal = np.nan, np.nan

    test_rows.append({
        "variable": var,
        "anova_F": f_stat,
        "anova_p": p_anova,
        "kruskal_H": h_stat,
        "kruskal_p": p_kruskal,
        "n_total": len(tmp),
    })

summary = pd.DataFrame(summary_rows)
tests = pd.DataFrame(test_rows)

# FDR correction
def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(np.nan_to_num(p, nan=1.0))
    ranked = np.empty(n)
    prev = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        idx = order[i]
        val = p[idx] * n / rank if not np.isnan(p[idx]) else np.nan
        if not np.isnan(val):
            prev = min(prev, val)
            ranked[idx] = prev
        else:
            ranked[idx] = np.nan

    return np.clip(ranked, 0, 1)

tests["anova_fdr_bh"] = bh_fdr(tests["anova_p"])
tests["kruskal_fdr_bh"] = bh_fdr(tests["kruskal_p"])

summary.to_csv(
    f"{OUT_DIR}/51_cluster_metabolic_profile_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

tests.to_csv(
    f"{OUT_DIR}/51_cluster_metabolic_anova_tests.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[SUMMARY]")
print(summary)

print("\n[TESTS]")
print(tests)

print("\n[SAVED]")
print(OUT_DIR)