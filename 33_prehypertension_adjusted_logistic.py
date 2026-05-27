# ============================================
# 33_prehypertension_adjusted_logistic.py
# ============================================

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm

from statsmodels.stats.multitest import multipletests

# --------------------------------------------
# PATH
# --------------------------------------------

INPUT = r"D:\precision_nutrition\FFQ\processed\analysis_cohort_semihealthy_foodgroups.csv"

OUT_DIR = r"D:\precision_nutrition\FFQ\results\33_prehypertension_adjusted_logistic"

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------
# LOAD
# --------------------------------------------

df = pd.read_csv(INPUT)

# --------------------------------------------
# PREHTN ONLY
# --------------------------------------------

sub = df[df["bp_group"].isin([
    "normal_bp",
    "prehypertension_clean"
])].copy()

sub["target"] = np.where(
    sub["bp_group"] == "prehypertension_clean",
    1,
    0
)

print("[GROUP]")
print(sub["bp_group"].value_counts())

# --------------------------------------------
# FOOD GROUPS
# --------------------------------------------

food_groups = [
    "fg_refined_grain_z",
    "fg_whole_grain_z",
    "fg_fastfood_z",
    "fg_meat_processed_z",
    "fg_fish_seafood_z",
    "fg_vegetable_z",
    "fg_kimchi_fermented_z",
    "fg_fruit_z",
    "fg_dairy_z",
    "fg_sweet_beverage_z",
    "fg_alcohol_z",
    "fg_coffee_tea_z"
]

# --------------------------------------------
# COVARIATES
# --------------------------------------------

covariates = [
    "age",
    "sex",
    "HE_BMI",
    "FQ_EN"
]

# --------------------------------------------
# NUMERIC
# --------------------------------------------

all_cols = food_groups + covariates + ["target"]

for c in all_cols:
    sub[c] = pd.to_numeric(sub[c], errors="coerce")

# --------------------------------------------
# RESULTS
# --------------------------------------------

rows = []

for fg in food_groups:

    print(f"\n[MODEL] {fg}")

    model_df = sub[
        ["target", fg] + covariates
    ].dropna()

    X = model_df[
        [fg] + covariates
    ]

    y = model_df["target"]

    # intercept
    X = sm.add_constant(X)

    model = sm.Logit(y, X)

    try:

        result = model.fit(disp=0)

        coef = result.params[fg]
        se = result.bse[fg]
        p = result.pvalues[fg]

        OR = np.exp(coef)

        CI_low = np.exp(coef - 1.96 * se)
        CI_high = np.exp(coef + 1.96 * se)

        rows.append({
            "food_group": fg,
            "OR": OR,
            "CI_low": CI_low,
            "CI_high": CI_high,
            "p_value": p,
            "coef": coef,
            "n": len(model_df)
        })

    except Exception as e:

        print(e)

# --------------------------------------------
# RESULT DF
# --------------------------------------------

res = pd.DataFrame(rows)

# FDR
res["fdr_bh"] = multipletests(
    res["p_value"],
    method="fdr_bh"
)[1]

# sort
res = res.sort_values(
    "OR",
    ascending=False
)

# --------------------------------------------
# SAVE
# --------------------------------------------

res.to_csv(
    f"{OUT_DIR}/33_adjusted_logistic_foodgroups.csv",
    index=False
)

print("\n[RESULT]")
print(res)

print("\n[SAVED]")
print(OUT_DIR)