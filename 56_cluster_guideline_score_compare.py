import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

BASE_DIR = r"D:\precision_nutrition\FFQ"

SCORE_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "55_guideline_diet_scores",
    "55_guideline_diet_scores_dataset.csv"
)

CLUSTER_INPUT = os.path.join(
    BASE_DIR,
    "results",
    "50_predm_latent_foodpattern_cluster",
    "50_predm_cluster_assigned_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "56_cluster_guideline_score_compare"
)
os.makedirs(OUT_DIR, exist_ok=True)

score_df = pd.read_csv(SCORE_INPUT, encoding="utf-8-sig", low_memory=False)
cluster_df = pd.read_csv(CLUSTER_INPUT, encoding="utf-8-sig", low_memory=False)

score_cols = [
    "score_aMED_proxy",
    "score_DASH_proxy",
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_RFS_proxy",
]

# merge
if "ID" in score_df.columns and "ID" in cluster_df.columns:
    df = cluster_df[["ID", "cluster"]].merge(
        score_df[["ID"] + score_cols],
        on="ID",
        how="left"
    )
else:
    print("[WARNING] ID 없음. row-order fallback 사용.")
    predm_score = score_df[score_df["glucose_group"] == "prediabetes_clean"].copy()
    predm_score = predm_score.reset_index(drop=True)
    cluster_df = cluster_df.reset_index(drop=True)
    df = pd.concat(
        [cluster_df[["cluster"]], predm_score[score_cols]],
        axis=1
    )

print("[INPUT]")
print(df.shape)
print(df["cluster"].value_counts().sort_index())

for c in score_cols + ["cluster"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# summary
summary_rows = []
test_rows = []
posthoc_rows = []

for score in score_cols:
    tmp = df[["cluster", score]].dropna()

    for cl in sorted(tmp["cluster"].unique()):
        x = tmp.loc[tmp["cluster"] == cl, score]
        summary_rows.append({
            "score": score,
            "cluster": int(cl),
            "n": len(x),
            "mean": x.mean(),
            "sd": x.std(),
            "median": x.median(),
            "q1": x.quantile(0.25),
            "q3": x.quantile(0.75),
        })

    groups = [
        tmp.loc[tmp["cluster"] == cl, score]
        for cl in sorted(tmp["cluster"].unique())
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
        groups=tmp["cluster"].astype(str),
        alpha=0.05
    )

    tukey_df = pd.DataFrame(
        tukey.summary().data[1:],
        columns=tukey.summary().data[0]
    )
    tukey_df["score"] = score
    posthoc_rows.append(tukey_df)

summary = pd.DataFrame(summary_rows)
tests = pd.DataFrame(test_rows)
posthoc = pd.concat(posthoc_rows, axis=0)

summary.to_csv(
    os.path.join(OUT_DIR, "56_cluster_guideline_score_summary.csv"),
    index=False,
    encoding="utf-8-sig"
)

tests.to_csv(
    os.path.join(OUT_DIR, "56_cluster_guideline_score_tests.csv"),
    index=False,
    encoding="utf-8-sig"
)

posthoc.to_csv(
    os.path.join(OUT_DIR, "56_cluster_guideline_score_posthoc.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[SUMMARY]")
print(summary)

print("\n[TESTS]")
print(tests)

print("\n[POSTHOC]")
print(posthoc)

print("\n[SAVED]")
print(OUT_DIR)