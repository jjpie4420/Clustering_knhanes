import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

INPUT = r"D:\precision_nutrition\FFQ\results\50_predm_latent_foodpattern_cluster\50_predm_cluster_assigned_dataset.csv"
FULL_INPUT = r"D:\precision_nutrition\FFQ\processed\analysis_cohort_semihealthy_foodgroups.csv"

OUT_DIR = r"D:\precision_nutrition\FFQ\results\52_predm_cluster_comorbidity_prevalence"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load cluster-assigned preDM dataset
# --------------------------------------------------
cl = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)
full = pd.read_csv(FULL_INPUT, encoding="utf-8-sig", low_memory=False)

print("[CLUSTER INPUT]")
print(cl.shape)

print("[FULL INPUT]")
print(full.shape)

# --------------------------------------------------
# Merge labels back if needed
# --------------------------------------------------
# 50번 저장 파일에 ID가 있으면 ID 기준 merge
# ID가 없으면 index 순서가 유지됐다고 보고 concat 방식 사용
# 가능하면 ID merge가 안전함
# --------------------------------------------------

label_cols = [
    "label_prehypertension_clean2",
    "label_borderline_lipid_clean",
    #"label_overweight_clean",
    "label_hypertension",
    "label_dyslipidemia",
    #"label_obesity",
    "bp_group",
    "lipid_group",
    #"bmi_group",
]

label_cols = [c for c in label_cols if c in full.columns]

if "ID" in cl.columns and "ID" in full.columns:
    use_cols = ["ID"] + label_cols
    df = cl.merge(
        full[use_cols],
        on="ID",
        how="left"
    )
else:
    print("[WARNING] ID column not found. Using row-order fallback.")
    full_predm = full[full["glucose_group"] == "prediabetes_clean"].copy()
    full_predm = full_predm.reset_index(drop=True)
    cl = cl.reset_index(drop=True)

    df = pd.concat(
        [cl, full_predm[label_cols].reset_index(drop=True)],
        axis=1
    )

print("[MERGED]")
print(df.shape)

print("\n[CLUSTER COUNTS]")
print(df["cluster"].value_counts().sort_index())

# --------------------------------------------------
# Define outcomes
# --------------------------------------------------

outcomes = {}

if "label_prehypertension_clean2" in df.columns:
    outcomes["prehypertension_clean"] = "label_prehypertension_clean2"

if "label_borderline_lipid_clean" in df.columns:
    outcomes["borderline_lipid_clean"] = "label_borderline_lipid_clean"

# if "label_overweight_clean" in df.columns:
#    outcomes["overweight_clean"] = "label_overweight_clean"

if "label_hypertension" in df.columns:
    outcomes["hypertension"] = "label_hypertension"

if "label_dyslipidemia" in df.columns:
    outcomes["dyslipidemia"] = "label_dyslipidemia"

#if "label_obesity" in df.columns:
#    outcomes["obesity"] = "label_obesity"

print("\n[OUTCOMES]")
print(outcomes)

# --------------------------------------------------
# Prevalence summary
# --------------------------------------------------

rows = []

for outcome_name, col in outcomes.items():
    df[col] = pd.to_numeric(df[col], errors="coerce")

    for cluster in sorted(df["cluster"].dropna().unique()):
        tmp = df[df["cluster"] == cluster]

        valid = tmp[col].dropna()
        n = len(valid)
        n_positive = int((valid == 1).sum())
        prevalence = n_positive / n if n > 0 else np.nan

        rows.append({
            "outcome": outcome_name,
            "cluster": int(cluster),
            "n": n,
            "n_positive": n_positive,
            "prevalence": prevalence,
            "prevalence_percent": prevalence * 100 if pd.notna(prevalence) else np.nan
        })

prev = pd.DataFrame(rows)

prev.to_csv(
    f"{OUT_DIR}/52_cluster_comorbidity_prevalence.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[PREVALENCE]")
print(prev)

# --------------------------------------------------
# Chi-square tests
# --------------------------------------------------

test_rows = []

for outcome_name, col in outcomes.items():
    tmp = df[["cluster", col]].dropna().copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp[tmp[col].isin([0, 1])]

    table = pd.crosstab(tmp["cluster"], tmp[col])

    if table.shape[0] >= 2 and table.shape[1] == 2:
        chi2, p, dof, expected = chi2_contingency(table)

        test_rows.append({
            "outcome": outcome_name,
            "chi2": chi2,
            "p_value": p,
            "dof": dof,
            "n_total": len(tmp)
        })

    else:
        test_rows.append({
            "outcome": outcome_name,
            "chi2": np.nan,
            "p_value": np.nan,
            "dof": np.nan,
            "n_total": len(tmp)
        })

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
        if np.isnan(p[idx]):
            ranked[idx] = np.nan
        else:
            val = p[idx] * n / rank
            prev = min(prev, val)
            ranked[idx] = prev

    return np.clip(ranked, 0, 1)

tests["fdr_bh"] = bh_fdr(tests["p_value"])

tests.to_csv(
    f"{OUT_DIR}/52_cluster_comorbidity_chisq_tests.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[CHI-SQUARE TESTS]")
print(tests)

# --------------------------------------------------
# Wide table for PPT
# --------------------------------------------------

wide = prev.pivot(
    index="outcome",
    columns="cluster",
    values="prevalence_percent"
).reset_index()

wide.columns = [
    "outcome"
] + [f"cluster_{int(c)}_prevalence_percent" for c in wide.columns[1:]]

wide = wide.merge(
    tests[["outcome", "p_value", "fdr_bh"]],
    on="outcome",
    how="left"
)

wide.to_csv(
    f"{OUT_DIR}/52_cluster_comorbidity_prevalence_wide.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[WIDE SUMMARY]")
print(wide)

print("\n[SAVED]")
print(OUT_DIR)