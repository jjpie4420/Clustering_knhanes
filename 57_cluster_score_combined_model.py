import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
    "57_cluster_score_combined_model"
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

outcome_cols = [
    "label_prehypertension_clean2",
    "label_borderline_lipid_clean",
    "label_overweight_clean",
    "label_hypertension",
    "label_dyslipidemia",
    "label_obesity",
]

covariates = [
    "age",
    "sex",
    "HE_BMI",
]

available_outcomes = [c for c in outcome_cols if c in score_df.columns]
available_covars = [c for c in covariates if c in score_df.columns]

if "ID" in score_df.columns and "ID" in cluster_df.columns:
    df = cluster_df[["ID", "cluster"]].merge(
        score_df[["ID"] + score_cols + available_outcomes + available_covars],
        on="ID",
        how="left"
    )
else:
    print("[WARNING] ID 없음. row-order fallback 사용.")
    predm_score = score_df[score_df["glucose_group"] == "prediabetes_clean"].copy()
    predm_score = predm_score.reset_index(drop=True)
    cluster_df = cluster_df.reset_index(drop=True)
    df = pd.concat(
        [
            cluster_df[["cluster"]],
            predm_score[score_cols + available_outcomes + available_covars]
        ],
        axis=1
    )

for c in ["cluster"] + score_cols + available_outcomes + available_covars:
    df[c] = pd.to_numeric(df[c], errors="coerce")

rows = []

for outcome in available_outcomes:
    for score in score_cols:

        tmp = df[
            ["cluster", score, outcome] + available_covars
        ].dropna()

        tmp = tmp[tmp[outcome].isin([0, 1])].copy()

        if tmp[outcome].nunique() < 2:
            continue

        # cluster dummy, C1 reference
        cluster_dummies = pd.get_dummies(
            tmp["cluster"].astype(int).astype(str),
            prefix="cluster",
            drop_first=True
        ).astype(float)

        X = pd.concat(
            [
                cluster_dummies,
                tmp[[score] + available_covars].astype(float)
            ],
            axis=1
        )

        X = sm.add_constant(X)
        y = tmp[outcome].astype(float)

        try:
            model = sm.Logit(y, X).fit(disp=0)

            for term in model.params.index:
                if term == "const":
                    continue

                coef = model.params[term]
                se = model.bse[term]
                p = model.pvalues[term]

                rows.append({
                    "outcome": outcome,
                    "score_model": score,
                    "term": term,
                    "OR": np.exp(coef),
                    "CI_low": np.exp(coef - 1.96 * se),
                    "CI_high": np.exp(coef + 1.96 * se),
                    "coef": coef,
                    "p_value": p,
                    "n": len(tmp)
                })

        except Exception as e:
            print(f"[ERROR] outcome={outcome}, score={score}: {e}")

res = pd.DataFrame(rows)

res.to_csv(
    os.path.join(OUT_DIR, "57_cluster_score_combined_model_results.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("[RESULT]")
print(res)

print("\n[SAVED]")
print(OUT_DIR)