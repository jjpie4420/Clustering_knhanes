import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR, "results", "55_guideline_diet_scores",
    "55_guideline_diet_scores_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR, "results", "61_foodgroup_score_combined_model"
)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

food_features = [
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

score_cols = [
    "score_aMED_proxy",
    "score_DASH_proxy",
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_RFS_proxy",
]

outcomes = {
    "preHTN": "label_prehypertension_clean2",
    "preDM": "label_prediabetes_clean2",
    "borderline_lipid": "label_borderline_lipid_clean"
}

covars = [
    "age",
    "sex",
    "HE_BMI",
]

covars = [c for c in covars if c in df.columns]

for c in food_features + score_cols + list(outcomes.values()) + covars:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

rows = []

def run_logit(tmp, y_col, x_cols, model_name, outcome_name):
    tmp = tmp[[y_col] + x_cols].dropna().copy()
    tmp = tmp[tmp[y_col].isin([0, 1])]

    if tmp[y_col].nunique() < 2 or len(tmp) < 500:
        return []

    X = tmp[x_cols].astype(float)
    y = tmp[y_col].astype(float)

    X = sm.add_constant(X)

    try:
        model = sm.Logit(y, X).fit(disp=0, maxiter=200)
    except Exception as e:
        print(f"[ERROR] {outcome_name} {model_name}: {e}")
        return []

    out = []

    for term in model.params.index:
        if term == "const":
            continue

        coef = model.params[term]
        se = model.bse[term]
        p = model.pvalues[term]

        out.append({
            "outcome": outcome_name,
            "model": model_name,
            "term": term,
            "OR": np.exp(coef),
            "CI_low": np.exp(coef - 1.96 * se),
            "CI_high": np.exp(coef + 1.96 * se),
            "coef": coef,
            "p_value": p,
            "n": len(tmp),
            "aic": model.aic,
            "bic": model.bic
        })

    return out

for outcome_name, y_col in outcomes.items():

    if y_col not in df.columns:
        continue

    print(f"\n[OUTCOME] {outcome_name}")

    base_covars = covars.copy()

    # Model A: food-group only
    rows.extend(
        run_logit(
            df,
            y_col,
            food_features + base_covars,
            "foodgroup_only",
            outcome_name
        )
    )

    # Model B: score only
    for score in score_cols:
        rows.extend(
            run_logit(
                df,
                y_col,
                [score] + base_covars,
                f"score_only_{score}",
                outcome_name
            )
        )

    # Model C: food group + each score
    for score in score_cols:
        rows.extend(
            run_logit(
                df,
                y_col,
                food_features + [score] + base_covars,
                f"foodgroup_plus_{score}",
                outcome_name
            )
        )

res = pd.DataFrame(rows)

res.to_csv(
    os.path.join(OUT_DIR, "61_combined_model_results.csv"),
    index=False,
    encoding="utf-8-sig"
)

# key summary: food feature changes after score adjustment
key_terms = food_features + score_cols

key = res[res["term"].isin(key_terms)].copy()

key.to_csv(
    os.path.join(OUT_DIR, "61_combined_model_key_terms.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[RESULT HEAD]")
print(res.head(50))

print("\n[SAVED]")
print(OUT_DIR)