# ============================================================
# 82_lipid_cluster_posthoc.py
#
# borderline lipid dietary subtype별 metabolic profile post-hoc
# - Tukey HSD
# - FDR correction
# ============================================================

import os
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "81_lipid_cluster_metabolic_anova",
    "81_lipid_cluster_metabolic_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "82_lipid_cluster_posthoc"
)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

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

metabolic_vars = [v for v in metabolic_vars if v in df.columns]

all_results = []

for var in metabolic_vars:

    tmp = df[["cluster", var]].dropna().copy()
    tmp["cluster"] = tmp["cluster"].astype(str)

    if tmp["cluster"].nunique() < 2:
        continue

    print(f"\n[POSTHOC] {var}")

    tukey = pairwise_tukeyhsd(
        endog=tmp[var],
        groups=tmp["cluster"],
        alpha=0.05
    )

    result_df = pd.DataFrame(
        tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    result_df["variable"] = var

    print(result_df)

    all_results.append(result_df)

final = pd.concat(all_results, axis=0)

for c in ["meandiff", "p-adj", "lower", "upper"]:
    final[c] = pd.to_numeric(final[c], errors="coerce")

final["fdr_bh"] = multipletests(
    final["p-adj"],
    method="fdr_bh"
)[1]

final["significant_fdr"] = final["fdr_bh"] < 0.05

final = final.sort_values(
    ["variable", "p-adj"]
)

save_path = os.path.join(
    OUT_DIR,
    "82_lipid_cluster_posthoc_tukey.csv"
)

final.to_csv(
    save_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n[FINAL RESULT]")
print(final)

print("\n[SAVED]")
print(save_path)