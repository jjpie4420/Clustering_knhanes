import os
import numpy as np
import pandas as pd

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

INPUT = r"D:\precision_nutrition\FFQ\results\50_predm_latent_foodpattern_cluster\50_predm_cluster_assigned_dataset.csv"

OUT_DIR = r"D:\precision_nutrition\FFQ\results\53_predm_cluster_posthoc"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

# -----------------------------
# Variables
# -----------------------------

vars_to_test = [
    "HE_BMI",
    "HE_wc",
    "HE_glu",
    "HE_HbA1c",
    "HE_TG",
    "HE_HDL_st2",
    "HE_sbp",
    "HE_dbp"
]

vars_to_test = [v for v in vars_to_test if v in df.columns]

# -----------------------------
# Tukey HSD
# -----------------------------

all_results = []

for var in vars_to_test:

    tmp = df[["cluster", var]].dropna().copy()

    tmp["cluster"] = tmp["cluster"].astype(str)

    print(f"\n[POSTHOC] {var}")

    try:
        tukey = pairwise_tukeyhsd(
            endog=tmp[var],
            groups=tmp["cluster"],
            alpha=0.05
        )

        result_df = pd.DataFrame(
            data=tukey.summary().data[1:],
            columns=tukey.summary().data[0]
        )

        result_df["variable"] = var

        all_results.append(result_df)

        print(result_df)

    except Exception as e:
        print(f"[ERROR] {var}: {e}")

# -----------------------------
# Merge
# -----------------------------

final = pd.concat(all_results, axis=0)

# numeric conversion
for c in ["meandiff", "p-adj", "lower", "upper"]:
    final[c] = pd.to_numeric(final[c], errors="coerce")

# FDR
final["fdr_bh"] = multipletests(
    final["p-adj"],
    method="fdr_bh"
)[1]

# significance label
final["significant_fdr"] = final["fdr_bh"] < 0.05

# sort
final = final.sort_values(
    ["variable", "p-adj"]
)

# save
save_path = f"{OUT_DIR}/53_cluster_posthoc_tukey.csv"

final.to_csv(
    save_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n[FINAL RESULT]")
print(final)

print("\n[SAVED]")
print(save_path)