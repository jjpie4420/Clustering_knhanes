import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR, "results", "55_guideline_diet_scores",
    "55_guideline_diet_scores_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR, "results", "58_semihealthy_burden_stratification"
)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

labels = [
    "label_prediabetes_clean2",
    "label_prehypertension_clean2",
    "label_borderline_lipid_clean"
]

labels = [c for c in labels if c in df.columns]

print("[LABELS USED]")
print(labels)

for c in labels:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

df["semihealthy_burden_count"] = df[labels].sum(axis=1)

df["semihealthy_burden_group"] = pd.cut(
    df["semihealthy_burden_count"],
    bins=[-0.1, 0.5, 1.5, 2.5, 10],
    labels=["0", "1", "2", "3+"]
)

score_cols = [
    "score_aMED_proxy",
    "score_DASH_proxy",
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_RFS_proxy",
]

food_features = [
    c for c in df.columns
    if c.startswith("fg_") and c.endswith("_z")
]

# --------------------------------------------------
# burden count summary
# --------------------------------------------------

burden_summary = (
    df["semihealthy_burden_group"]
    .value_counts(dropna=False)
    .sort_index()
    .reset_index()
)

burden_summary.columns = ["burden_group", "n"]

print("\n[BURDEN SUMMARY]")
print(burden_summary)

burden_summary.to_csv(
    os.path.join(OUT_DIR, "58_burden_group_counts.csv"),
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# score trend by burden group
# --------------------------------------------------

summary_rows = []
test_rows = []
posthoc_rows = []

for score in score_cols:
    if score not in df.columns:
        continue

    tmp = df[["semihealthy_burden_group", score]].dropna().copy()

    for g in ["0", "1", "2", "3+"]:
        x = tmp.loc[tmp["semihealthy_burden_group"].astype(str) == g, score]

        summary_rows.append({
            "score": score,
            "burden_group": g,
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "median": x.median(),
            "q1": x.quantile(0.25),
            "q3": x.quantile(0.75),
        })

    groups = [
        tmp.loc[tmp["semihealthy_burden_group"].astype(str) == g, score]
        for g in ["0", "1", "2", "3+"]
        if len(tmp.loc[tmp["semihealthy_burden_group"].astype(str) == g, score]) > 0
    ]

    f, p_anova = f_oneway(*groups)
    h, p_kw = kruskal(*groups)

    test_rows.append({
        "score": score,
        "anova_F": f,
        "anova_p": p_anova,
        "kruskal_H": h,
        "kruskal_p": p_kw,
        "n_total": len(tmp)
    })

    tukey = pairwise_tukeyhsd(
        endog=tmp[score],
        groups=tmp["semihealthy_burden_group"].astype(str),
        alpha=0.05
    )

    tukey_df = pd.DataFrame(
        tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )
    tukey_df["score"] = score
    posthoc_rows.append(tukey_df)

score_summary = pd.DataFrame(summary_rows)
score_tests = pd.DataFrame(test_rows)
score_posthoc = pd.concat(posthoc_rows, axis=0)

score_summary.to_csv(
    os.path.join(OUT_DIR, "58_burden_guideline_score_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

score_tests.to_csv(
    os.path.join(OUT_DIR, "58_burden_guideline_score_tests.csv"),
    index=False,
    encoding="utf-8-sig"
)

score_posthoc.to_csv(
    os.path.join(OUT_DIR, "58_burden_guideline_score_posthoc.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SCORE SUMMARY]")
print(score_summary)

print("\n[SCORE TESTS]")
print(score_tests)

# --------------------------------------------------
# food-group profile by burden group
# --------------------------------------------------

food_profile = (
    df.groupby("semihealthy_burden_group", observed=False)[food_features]
    .mean()
    .T
)

food_profile.to_csv(
    os.path.join(OUT_DIR, "58_burden_foodgroup_profile.csv"),
    encoding="utf-8-sig"
)

print("\n[FOOD PROFILE]")
print(food_profile)

# --------------------------------------------------
# specific phenotype combination table
# --------------------------------------------------

combo_cols = labels.copy()

df["burden_combo"] = df[combo_cols].astype(int).astype(str).agg("_".join, axis=1)

combo_summary = (
    df.groupby("burden_combo")
    .size()
    .reset_index(name="n")
    .sort_values("n", ascending=False)
)

combo_summary.to_csv(
    os.path.join(OUT_DIR, "58_specific_burden_combination_counts.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[COMBO SUMMARY]")
print(combo_summary.head(20))

# save full
df.to_csv(
    os.path.join(OUT_DIR, "58_dataset_with_burden.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SAVED]")
print(OUT_DIR)