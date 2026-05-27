import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

INPUT = r"D:\precision_nutrition\FFQ\processed\analysis_cohort_semihealthy_foodgroups.csv"
OUT_DIR = r"D:\precision_nutrition\FFQ\results\45_prediabetes_adjusted_logistic"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

sub = df[df["glucose_group"].isin([
    "normal_glucose",
    "prediabetes_clean"
])].copy()

sub["target"] = np.where(
    sub["glucose_group"] == "prediabetes_clean",
    1,
    0
)

print("[GROUP]")
print(sub["glucose_group"].value_counts())

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
    "fg_coffee_tea_z",
]

covariates = [
    "age",
    "sex",
    "HE_BMI",
    "FQ_EN",
]

all_cols = food_groups + covariates + ["target"]

for c in all_cols:
    sub[c] = pd.to_numeric(sub[c], errors="coerce")

rows = []

for fg in food_groups:
    print(f"\n[MODEL] {fg}")

    tmp = sub[["target", fg] + covariates].dropna()

    X = tmp[[fg] + covariates]
    y = tmp["target"]

    X = sm.add_constant(X)

    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0)

        coef = result.params[fg]
        se = result.bse[fg]
        p = result.pvalues[fg]

        rows.append({
            "food_group": fg,
            "OR": np.exp(coef),
            "CI_low": np.exp(coef - 1.96 * se),
            "CI_high": np.exp(coef + 1.96 * se),
            "coef": coef,
            "p_value": p,
            "n": len(tmp),
        })

    except Exception as e:
        print(f"[ERROR] {fg}: {e}")

res = pd.DataFrame(rows)

res["fdr_bh"] = multipletests(
    res["p_value"],
    method="fdr_bh"
)[1]

res = res.sort_values("OR", ascending=False)

out_file = f"{OUT_DIR}/45_prediabetes_adjusted_logistic.csv"

res.to_csv(
    out_file,
    index=False,
    encoding="utf-8-sig"
)

print("\n[RESULT]")
print(res)

print("\n[SAVED]")
print(out_file)