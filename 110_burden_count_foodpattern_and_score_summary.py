# ============================================================
# 110_burden_count_foodpattern_and_score_summary.py
#
# Burden count analysis:
# 0 / 1 / 2 / 3 predisease burden
#
# Outputs:
# - N summary
# - food-group profile heatmap
# - dietary score summary + ANOVA/posthoc
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "94_mutually_exclusive_predisease_groups",
    "94_dataset_with_exclusive_predisease_groups.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "110_burden_count_foodpattern_and_score_summary"
)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

# ============================================================
# Burden count
# ============================================================

if "predisease_burden_n" not in df.columns:
    raise ValueError("predisease_burden_n column not found. Run 94 first.")

df["predisease_burden_n"] = pd.to_numeric(
    df["predisease_burden_n"],
    errors="coerce"
)

df = df[df["predisease_burden_n"].isin([0, 1, 2, 3])].copy()

df["burden_group"] = df["predisease_burden_n"].map({
    0: "0 burden",
    1: "1 burden",
    2: "2 burdens",
    3: "3 burdens",
})

burden_order = ["0 burden", "1 burden", "2 burdens", "3 burdens"]

print("\n[BURDEN COUNTS]")
print(df["burden_group"].value_counts().reindex(burden_order))

# ============================================================
# Variables
# ============================================================

food_vars = [
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

food_labels = {
    "fg_refined_grain_z": "Refined grain",
    "fg_whole_grain_z": "Whole grain",
    "fg_fastfood_z": "Fast food / sweets",
    "fg_meat_processed_z": "Meat / processed meat",
    "fg_fish_seafood_z": "Fish / seafood",
    "fg_vegetable_z": "Vegetables",
    "fg_kimchi_fermented_z": "Kimchi / fermented veg",
    "fg_fruit_z": "Fruit",
    "fg_dairy_z": "Dairy",
    "fg_sweet_beverage_z": "Sweet beverage",
    "fg_alcohol_z": "Alcohol",
    "fg_coffee_tea_z": "Coffee / tea",
}

score_vars = [
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_DASH_proxy",
    "score_aMED_proxy",
    "score_RFS_proxy",
]

score_labels = {
    "score_AHEI_proxy": "AHEI",
    "score_HEI_proxy": "HEI",
    "score_DASH_proxy": "DASH",
    "score_aMED_proxy": "aMED",
    "score_RFS_proxy": "RFS",
}

metabolic_vars = {
    "HE_BMI": "BMI",
    "HE_wc": "Waist",
    "HE_glu": "Glucose",
    "HE_HbA1c": "HbA1c",
    "HE_TG": "TG",
    "HE_HDL_st2": "HDL-C",
    "HE_sbp": "SBP",
    "HE_dbp": "DBP",
}

food_vars = [v for v in food_vars if v in df.columns]
score_vars = [v for v in score_vars if v in df.columns]
metabolic_vars = {k: v for k, v in metabolic_vars.items() if k in df.columns}

for c in food_vars + score_vars + list(metabolic_vars.keys()):
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ============================================================
# N summary
# ============================================================

n_summary = (
    df["burden_group"]
    .value_counts()
    .reindex(burden_order)
    .reset_index()
)

n_summary.columns = ["burden_group", "n"]
n_summary["percent"] = n_summary["n"] / n_summary["n"].sum() * 100

n_summary.to_csv(
    os.path.join(OUT_DIR, "110_burden_count_n_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[N SUMMARY]")
print(n_summary)

# ============================================================
# Food profile by burden count
# ============================================================

food_profile = (
    df.groupby("burden_group")[food_vars]
    .mean()
    .reindex(burden_order)
)

food_profile.columns = [food_labels.get(c, c) for c in food_profile.columns]

food_profile.to_csv(
    os.path.join(OUT_DIR, "110_burden_count_food_profile.csv"),
    encoding="utf-8-sig"
)

# z-score across burden groups per food variable
food_z = food_profile.copy()

for col in food_z.columns:
    sd = food_z[col].std()
    if sd == 0 or pd.isna(sd):
        food_z[col] = 0
    else:
        food_z[col] = (food_z[col] - food_z[col].mean()) / sd

food_z.to_csv(
    os.path.join(OUT_DIR, "110_burden_count_food_profile_zscore.csv"),
    encoding="utf-8-sig"
)

plt.figure(figsize=(11, 4.8))

sns.heatmap(
    food_z,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title("Food-group profile by predisease burden count")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "110_burden_count_food_profile_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Dietary score summary + tests
# ============================================================

summary_rows = []
test_rows = []
posthoc_rows = []

for var in score_vars:

    label = score_labels.get(var, var)

    tmp = df[["burden_group", var]].dropna().copy()

    for g in burden_order:
        x = tmp.loc[tmp["burden_group"] == g, var]

        summary_rows.append({
            "variable": var,
            "score": label,
            "burden_group": g,
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "sem": x.std() / np.sqrt(len(x)) if len(x) > 0 else np.nan,
            "median": x.median(),
            "q1": x.quantile(0.25),
            "q3": x.quantile(0.75),
        })

    groups = [
        tmp.loc[tmp["burden_group"] == g, var]
        for g in burden_order
        if len(tmp.loc[tmp["burden_group"] == g, var]) > 1
    ]

    F, p_anova = f_oneway(*groups)
    H, p_kw = kruskal(*groups)

    test_rows.append({
        "variable": var,
        "score": label,
        "anova_F": F,
        "anova_p": p_anova,
        "kruskal_H": H,
        "kruskal_p": p_kw,
    })

    tukey = pairwise_tukeyhsd(
        endog=tmp[var],
        groups=tmp["burden_group"],
        alpha=0.05
    )

    tukey_df = pd.DataFrame(
        tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )

    tukey_df["variable"] = var
    tukey_df["score"] = label

    posthoc_rows.append(tukey_df)

score_summary = pd.DataFrame(summary_rows)
score_tests = pd.DataFrame(test_rows)
score_posthoc = pd.concat(posthoc_rows, axis=0, ignore_index=True)

score_tests["anova_fdr_bh"] = multipletests(
    score_tests["anova_p"].fillna(1),
    method="fdr_bh"
)[1]

score_tests["kruskal_fdr_bh"] = multipletests(
    score_tests["kruskal_p"].fillna(1),
    method="fdr_bh"
)[1]

score_posthoc["p-adj"] = pd.to_numeric(score_posthoc["p-adj"], errors="coerce")
score_posthoc["fdr_bh"] = multipletests(
    score_posthoc["p-adj"].fillna(1),
    method="fdr_bh"
)[1]

score_posthoc["significant_fdr"] = score_posthoc["fdr_bh"] < 0.05

score_summary.to_csv(
    os.path.join(OUT_DIR, "110_burden_count_score_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

score_tests.to_csv(
    os.path.join(OUT_DIR, "110_burden_count_score_tests.csv"),
    index=False,
    encoding="utf-8-sig"
)

score_posthoc.to_csv(
    os.path.join(OUT_DIR, "110_burden_count_score_posthoc.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SCORE TESTS]")
print(score_tests)

# ============================================================
# Score heatmap
# ============================================================

score_mean = (
    score_summary
    .pivot(index="burden_group", columns="score", values="mean")
    .reindex(burden_order)
)

score_z = score_mean.copy()

for col in score_z.columns:
    sd = score_z[col].std()
    if sd == 0 or pd.isna(sd):
        score_z[col] = 0
    else:
        score_z[col] = (score_z[col] - score_z[col].mean()) / sd

plt.figure(figsize=(7, 4.8))

sns.heatmap(
    score_z,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title("Dietary quality score by predisease burden count")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "110_burden_count_diet_score_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Score barplot
# ============================================================

plt.figure(figsize=(9, 5))

plot_df = score_summary.copy()

sns.barplot(
    data=plot_df,
    x="score",
    y="mean",
    hue="burden_group",
    hue_order=burden_order
)

plt.ylabel("Mean dietary quality score")
plt.xlabel("")
plt.title("Dietary quality scores by predisease burden count")
plt.legend(title="Burden count")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "110_burden_count_diet_score_barplot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Metabolic profile by burden count
# ============================================================

met_summary_rows = []

for var, label in metabolic_vars.items():

    tmp = df[["burden_group", var]].dropna().copy()

    for g in burden_order:
        x = tmp.loc[tmp["burden_group"] == g, var]

        met_summary_rows.append({
            "variable": var,
            "label": label,
            "burden_group": g,
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "median": x.median(),
            "q1": x.quantile(0.25),
            "q3": x.quantile(0.75),
        })

met_summary = pd.DataFrame(met_summary_rows)

met_summary.to_csv(
    os.path.join(OUT_DIR, "110_burden_count_metabolic_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

met_mean = (
    met_summary
    .pivot(index="burden_group", columns="label", values="mean")
    .reindex(burden_order)
)

met_z = met_mean.copy()

for col in met_z.columns:
    sd = met_z[col].std()
    if sd == 0 or pd.isna(sd):
        met_z[col] = 0
    else:
        met_z[col] = (met_z[col] - met_z[col].mean()) / sd

plt.figure(figsize=(9, 4.8))

sns.heatmap(
    met_z,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title("Metabolic profile by predisease burden count")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()

plt.savefig(
    os.path.join(OUT_DIR, "110_burden_count_metabolic_profile_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n[SAVED]")
print(OUT_DIR)