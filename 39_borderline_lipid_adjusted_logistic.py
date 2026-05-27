import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.multitest import multipletests

INPUT = r"D:\precision_nutrition\FFQ\processed\analysis_cohort_semihealthy_foodgroups.csv"

OUT_DIR = r"D:\precision_nutrition\FFQ\results\39_borderline_lipid_adjusted_logistic"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

# --------------------------------------------------
# BORDERLINE LIPID CLEAN
# --------------------------------------------------

sub = df[df["lipid_group"].isin([
    "normal_lipid",
    "borderline_lipid_clean"
])].copy()

sub["target"] = np.where(
    sub["lipid_group"] == "borderline_lipid_clean",
    1,
    0
)

print("[GROUP]")
print(sub["lipid_group"].value_counts())

# --------------------------------------------------
# FOOD GROUPS
# --------------------------------------------------

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

# --------------------------------------------------
# COVARIATES
# --------------------------------------------------

covariates = [
    "age",
    "sex",
    "HE_BMI",
    "FQ_EN"
]

# --------------------------------------------------
# NUMERIC
# --------------------------------------------------

all_cols = food_groups + covariates + ["target"]

for c in all_cols:
    sub[c] = pd.to_numeric(sub[c], errors="coerce")

# --------------------------------------------------
# RUN LOGISTIC
# --------------------------------------------------

rows = []

for fg in food_groups:

    print(f"\n[MODEL] {fg}")

    tmp = sub[
        ["target", fg] + covariates
    ].dropna()

    X = tmp[[fg] + covariates]
    y = tmp["target"]

    X = sm.add_constant(X)

    try:

        model = sm.Logit(y, X)

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
            "coef": coef,
            "p_value": p,
            "n": len(tmp)
        })

    except Exception as e:

        print(e)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

res = pd.DataFrame(rows)

res["fdr_bh"] = multipletests(
    res["p_value"],
    method="fdr_bh"
)[1]

res = res.sort_values(
    "OR",
    ascending=False
)

res.to_csv(
    f"{OUT_DIR}/39_borderline_lipid_adjusted_logistic.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[RESULT]")
print(res)

print("\n[SAVED]")
print(OUT_DIR)