# ============================================================
# 112_within_burden_adjusted_metabolic_models.py
#
# Same burden-combination 내부에서
# food subtype별 metabolic severity 차이 평가
#
# Input:
# 111에서 생성한 group별 food_cluster_dataset.csv
#
# Outputs:
# - descriptive summary
# - ANOVA / Kruskal tests
# - Tukey posthoc
# - adjusted OLS results
# - metabolic severity heatmap
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway, kruskal, zscore
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# ============================================================
# PATH
# ============================================================

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT_FILES = {
    "preDM_preHTN": os.path.join(
        BASE_DIR, "results", "111_burden_combination_food_clustering",
        "preDM_preHTN", "111_preDM_preHTN_food_cluster_dataset.csv"
    ),
    "preDM_lipid": os.path.join(
        BASE_DIR, "results", "111_burden_combination_food_clustering",
        "preDM_lipid", "111_preDM_lipid_food_cluster_dataset.csv"
    ),
    "preHTN_lipid": os.path.join(
        BASE_DIR, "results", "111_burden_combination_food_clustering",
        "preHTN_lipid", "111_preHTN_lipid_food_cluster_dataset.csv"
    ),
    "triple_burden": os.path.join(
        BASE_DIR, "results", "111_burden_combination_food_clustering",
        "triple_burden", "111_triple_burden_food_cluster_dataset.csv"
    ),
}

OUTDIR = Path(BASE_DIR) / "results" / "112_within_burden_adjusted_models"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# VARIABLES
# ============================================================

condition_list = [
    "preDM_preHTN",
    "preDM_lipid",
    "preHTN_lipid",
    "triple_burden",
]

metabolic_vars = {
    "BMI": "HE_BMI",
    "Waist": "HE_wc",
    "Glucose": "HE_glu",
    "HbA1c": "HE_HbA1c",
    "TG": "HE_TG",
    "HDL-C": "HE_HDL_st2",
    "SBP": "HE_sbp",
    "DBP": "HE_dbp",
}

# ============================================================
# HELPERS
# ============================================================

def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fdr_adjust(df, p_col, out_col):
    if df.empty or p_col not in df.columns:
        return df
    df[out_col] = multipletests(df[p_col].fillna(1), method="fdr_bh")[1]
    return df


def make_z_heatmap(summary_df, condition, outdir):
    heat = summary_df.pivot(
        index="food_subtype",
        columns="outcome",
        values="mean"
    )

    # row order: 있는 것만 유지
    preferred_order = [
        "Alcohol/beverage",
        "Refined/processed",
        "Prudent/high-quality",
        "Mixed/intermediate",
    ]
    order = [x for x in preferred_order if x in heat.index]
    remain = [x for x in heat.index if x not in order]
    heat = heat.loc[order + remain]

    heat_z = heat.copy()

    for col in heat_z.columns:
        sd = heat_z[col].std()
        if pd.isna(sd) or sd == 0:
            heat_z[col] = 0
        else:
            heat_z[col] = (heat_z[col] - heat_z[col].mean()) / sd

    heat.to_csv(
        outdir / f"112_{condition}_metabolic_profile_raw.csv",
        encoding="utf-8-sig"
    )
    heat_z.to_csv(
        outdir / f"112_{condition}_metabolic_profile_zscore.csv",
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(9, 5.2))
    sns.heatmap(
        heat_z,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title(f"Metabolic severity across food subtypes\n{condition}")
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(
        outdir / f"112_{condition}_metabolic_severity_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# MAIN
# ============================================================

all_summary = []
all_tests = []
all_posthoc = []
all_adj = []

for condition in condition_list:

    print("\n" + "=" * 80)
    print(condition)

    input_file = INPUT_FILES[condition]

    if not os.path.exists(input_file):
        print(f"[SKIP] file not found: {input_file}")
        continue

    sub = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)

    print("[INPUT]")
    print(sub.shape)

    cluster_col = "food_cluster"
    subtype_col = "food_subtype"

    if cluster_col not in sub.columns:
        raise ValueError(f"{condition}: food_cluster column not found")
    if subtype_col not in sub.columns:
        raise ValueError(f"{condition}: food_subtype column not found")

    available_outcomes = {
        label: col
        for label, col in metabolic_vars.items()
        if col in sub.columns
    }

    needed = [cluster_col, subtype_col, "age", "sex"] + list(available_outcomes.values())
    existing_needed = [c for c in needed if c in sub.columns]

    sub = safe_numeric(sub, list(available_outcomes.values()) + ["age", "sex"])
    sub = sub.dropna(subset=[cluster_col, subtype_col])

    print("[CLUSTER COUNTS]")
    print(sub[subtype_col].value_counts())

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    for outcome_label, outcome_col in available_outcomes.items():

        tmp = sub[[cluster_col, subtype_col, outcome_col]].copy()
        tmp[outcome_col] = pd.to_numeric(tmp[outcome_col], errors="coerce")
        tmp = tmp.dropna()

        for subtype in sorted(tmp[subtype_col].unique()):

            vals = tmp.loc[tmp[subtype_col] == subtype, outcome_col]

            summary_rows.append({
                "condition": condition,
                "outcome": outcome_label,
                "outcome_col": outcome_col,
                "food_subtype": subtype,
                "n": len(vals),
                "mean": vals.mean(),
                "sd": vals.std(),
                "median": vals.median(),
                "q1": vals.quantile(0.25),
                "q3": vals.quantile(0.75),
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTDIR / f"112_{condition}_metabolic_summary_by_food_subtype.csv",
        index=False,
        encoding="utf-8-sig"
    )

    all_summary.append(summary_df)

    # heatmap
    make_z_heatmap(summary_df, condition, OUTDIR)

    # ========================================================
    # ANOVA / KRUSKAL + POSTHOC
    # ========================================================

    test_rows = []
    posthoc_rows = []

    for outcome_label, outcome_col in available_outcomes.items():

        tmp = sub[[subtype_col, outcome_col]].copy()
        tmp[outcome_col] = pd.to_numeric(tmp[outcome_col], errors="coerce")
        tmp = tmp.dropna()

        groups = [
            tmp.loc[tmp[subtype_col] == g, outcome_col].values
            for g in sorted(tmp[subtype_col].unique())
        ]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            continue

        try:
            F, p_anova = f_oneway(*groups)
        except Exception:
            F, p_anova = np.nan, np.nan

        try:
            H, p_kw = kruskal(*groups)
        except Exception:
            H, p_kw = np.nan, np.nan

        test_rows.append({
            "condition": condition,
            "outcome": outcome_label,
            "anova_F": F,
            "anova_p": p_anova,
            "kruskal_H": H,
            "kruskal_p": p_kw,
        })

        try:
            tukey = pairwise_tukeyhsd(
                endog=tmp[outcome_col],
                groups=tmp[subtype_col],
                alpha=0.05
            )

            tukey_df = pd.DataFrame(
                tukey.summary().data[1:],
                columns=tukey.summary().data[0]
            )

            tukey_df["condition"] = condition
            tukey_df["outcome"] = outcome_label
            posthoc_rows.append(tukey_df)

        except Exception:
            pass

    test_df = pd.DataFrame(test_rows)
    test_df = fdr_adjust(test_df, "anova_p", "anova_fdr_bh")
    test_df = fdr_adjust(test_df, "kruskal_p", "kruskal_fdr_bh")

    test_df.to_csv(
        OUTDIR / f"112_{condition}_metabolic_tests.csv",
        index=False,
        encoding="utf-8-sig"
    )

    all_tests.append(test_df)

    if len(posthoc_rows) > 0:
        posthoc_df = pd.concat(posthoc_rows, ignore_index=True)
        if "p-adj" in posthoc_df.columns:
            posthoc_df["p-adj"] = pd.to_numeric(posthoc_df["p-adj"], errors="coerce")
            posthoc_df = fdr_adjust(posthoc_df, "p-adj", "posthoc_fdr_bh")
            posthoc_df["significant_fdr"] = posthoc_df["posthoc_fdr_bh"] < 0.05
    else:
        posthoc_df = pd.DataFrame()

    posthoc_df.to_csv(
        OUTDIR / f"112_{condition}_metabolic_posthoc.csv",
        index=False,
        encoding="utf-8-sig"
    )

    all_posthoc.append(posthoc_df)

    # ========================================================
    # ADJUSTED OLS
    # outcome ~ C(food_subtype) + age + sex + BMI
    # BMI/Waist outcome: BMI covariate 제외
    # ========================================================

    adj_rows = []

    for outcome_label, outcome_col in available_outcomes.items():

        covars = []

        if "age" in sub.columns:
            covars.append("age")
        if "sex" in sub.columns:
            covars.append("C(sex)")

        # BMI/Waist outcome에서는 HE_BMI 보정 제외
        if outcome_label not in ["BMI", "Waist"] and "HE_BMI" in sub.columns:
            covars.append("HE_BMI")

        model_cols = [outcome_col, subtype_col] + ["age", "sex"]
        if "HE_BMI" in covars:
            model_cols.append("HE_BMI")
        model_cols = [c for c in model_cols if c in sub.columns]

        tmp = sub[model_cols].copy()
        tmp[outcome_col] = pd.to_numeric(tmp[outcome_col], errors="coerce")
        if "age" in tmp.columns:
            tmp["age"] = pd.to_numeric(tmp["age"], errors="coerce")
        if "HE_BMI" in tmp.columns:
            tmp["HE_BMI"] = pd.to_numeric(tmp["HE_BMI"], errors="coerce")

        tmp = tmp.dropna()

        if tmp[subtype_col].nunique() < 2:
            continue

        formula = f"{outcome_col} ~ C({subtype_col})"
        if covars:
            formula += " + " + " + ".join(covars)

        try:
            model = ols(formula, data=tmp).fit(cov_type="HC3")
            coef = model.summary2().tables[1].reset_index()
            coef = coef.rename(columns={"index": "term"})
            coef["condition"] = condition
            coef["outcome"] = outcome_label
            coef["n"] = int(model.nobs)
            coef["r_squared"] = model.rsquared
            coef["adj_r_squared"] = model.rsquared_adj
            adj_rows.append(coef)
        except Exception as e:
            print(f"[ADJ MODEL ERROR] {condition} {outcome_label}: {e}")

    if len(adj_rows) > 0:
        adj_df = pd.concat(adj_rows, ignore_index=True)

        # subtype term만 따로 FDR 보정
        adj_df["is_subtype_term"] = adj_df["term"].astype(str).str.contains(f"C\\({subtype_col}\\)")
        subtype_terms = adj_df[adj_df["is_subtype_term"]].copy()

        if not subtype_terms.empty and "P>|z|" in subtype_terms.columns:
            subtype_terms["coef_fdr_bh"] = multipletests(
                subtype_terms["P>|z|"].fillna(1),
                method="fdr_bh"
            )[1]
            adj_df = adj_df.merge(
                subtype_terms[["condition", "outcome", "term", "coef_fdr_bh"]],
                on=["condition", "outcome", "term"],
                how="left"
            )
    else:
        adj_df = pd.DataFrame()

    adj_df.to_csv(
        OUTDIR / f"112_{condition}_adjusted_ols_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    all_adj.append(adj_df)

    print("[TESTS]")
    print(test_df)
    print("[SAVED CONDITION]")
    print(condition)

# ============================================================
# MERGED OUTPUTS
# ============================================================

if all_summary:
    pd.concat(all_summary, ignore_index=True).to_csv(
        OUTDIR / "112_ALL_metabolic_summary_by_food_subtype.csv",
        index=False,
        encoding="utf-8-sig"
    )

if all_tests:
    pd.concat(all_tests, ignore_index=True).to_csv(
        OUTDIR / "112_ALL_metabolic_tests.csv",
        index=False,
        encoding="utf-8-sig"
    )

if all_posthoc:
    pd.concat(all_posthoc, ignore_index=True).to_csv(
        OUTDIR / "112_ALL_metabolic_posthoc.csv",
        index=False,
        encoding="utf-8-sig"
    )

if all_adj:
    pd.concat(all_adj, ignore_index=True).to_csv(
        OUTDIR / "112_ALL_adjusted_ols_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("\n[SAVED]")
print(OUTDIR)