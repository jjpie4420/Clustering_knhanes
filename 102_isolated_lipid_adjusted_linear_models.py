# ============================================================
# 102_isolated_lipid_adjusted_linear_models.py
#
# Adjusted validation:
# Do dietary subtypes predict metabolic phenotype
# within isolated borderline lipid group?
#
# Linear model:
# outcome ~ C(subtype) + age + sex + HE_BMI
# ============================================================

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "95_isolated_lipid_latent_foodpattern_cluster",
    "95_isolated_lipid_cluster_assigned_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "102_isolated_lipid_adjusted_linear_models"
)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

# ============================================================
# 변수 설정
# ============================================================

outcomes = {
    "HE_TG": "TG",
    "HE_HDL_st2": "HDL-C",
    "HE_glu": "Glucose",
    "HE_HbA1c": "HbA1c",
    "HE_sbp": "SBP",
    "HE_dbp": "DBP",
}

covariates = ["age", "sex", "HE_BMI"]

# subtype 기준군 설정
# Cluster 1 = Prudent/high-quality
# Cluster 2 = Refined/beverage

df = df[df["cluster"].notna()].copy()
df["cluster"] = df["cluster"].astype(int)

df["subtype"] = df["cluster"].map({
    1: "Prudent_high_quality",
    2: "Refined_beverage"
})

df["subtype"] = pd.Categorical(
    df["subtype"],
    categories=["Prudent_high_quality", "Refined_beverage"]
)

# ============================================================
# 숫자형 변환
# ============================================================

for col in list(outcomes.keys()) + covariates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        print(f"[MISSING] {col}")

# sex 처리
# 이미 1/2 또는 0/1이면 그대로 categorical로 사용
df["sex"] = df["sex"].astype("category")

# ============================================================
# 모델 실행
# ============================================================

rows = []

for outcome, outcome_label in outcomes.items():

    if outcome not in df.columns:
        continue

    use_cols = [outcome, "subtype"] + covariates

    tmp = df[use_cols].dropna().copy()

    print("\n" + "=" * 70)
    print(f"[OUTCOME] {outcome_label}")
    print(tmp["subtype"].value_counts())

    formula = f"{outcome} ~ C(subtype, Treatment(reference='Prudent_high_quality')) + age + C(sex) + HE_BMI"

    model = smf.ols(
        formula=formula,
        data=tmp
    ).fit(cov_type="HC3")

    print(model.summary())

    term = "C(subtype, Treatment(reference='Prudent_high_quality'))[T.Refined_beverage]"

    if term in model.params.index:

        rows.append({
            "outcome": outcome,
            "outcome_label": outcome_label,
            "n": int(model.nobs),
            "reference": "Prudent/high-quality",
            "comparison": "Refined/beverage",
            "beta_refined_vs_prudent": model.params[term],
            "se": model.bse[term],
            "p_value": model.pvalues[term],
            "ci_lower": model.conf_int().loc[term, 0],
            "ci_upper": model.conf_int().loc[term, 1],
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
        })

results = pd.DataFrame(rows)

results["fdr_bh"] = multipletests(
    results["p_value"].fillna(1),
    method="fdr_bh"
)[1]

results["significant_fdr"] = results["fdr_bh"] < 0.05

# ============================================================
# 저장
# ============================================================

results.to_csv(
    os.path.join(
        OUT_DIR,
        "102_isolated_lipid_adjusted_linear_model_results.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n[FINAL RESULTS]")
print(results)

print("\n[SAVED]")
print(OUT_DIR)