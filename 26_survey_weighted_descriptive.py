import numpy as np
import pandas as pd
from config import PROCESSED_DIR, RESULT_DIR

INPUT = PROCESSED_DIR / "analysis_cohort_energy_adjusted.csv"

OUT_OVERALL = RESULT_DIR / "26_weighted_overall_descriptive.csv"
OUT_TARGET = RESULT_DIR / "26_weighted_target_prevalence.csv"
OUT_BY_SEX = RESULT_DIR / "26_weighted_target_by_sex.csv"

WEIGHT_CANDIDATES = [
    "wt_ntr",
    "wt_tot",
    "wt_hs",
    "wt_itvex",
]

TARGETS = {
    "diabetes": "label_diabetes",
    "diabetes_untreated": "label_diabetes_untreated",
    "prediabetes_clean": "label_prediabetes_clean",
    "hypertension": "label_hypertension",
    "hypertension_untreated": "label_hypertension_untreated",
    "prehypertension_clean": "label_prehypertension_clean",
    "dyslipidemia": "label_dyslipidemia",
    "dyslipidemia_untreated": "label_dyslipidemia_untreated",
    "obesity": "label_obesity",
}

CONTINUOUS_VARS = [
    "age",
    "HE_BMI",
    "HE_wc",
    "HE_sbp",
    "HE_dbp",
    "HE_glu",
    "HE_HbA1c",
    "HE_chol",
    "HE_HDL_st2",
    "HE_TG",
    "FQ_EN",
    "RC_EN",
    "FQ_PROT",
    "RC_PROT",
    "FQ_FAT",
    "RC_FAT",
    "FQ_CHO",
    "RC_CHO",
    "FQ_TDF",
    "RC_TDF",
    "FQ_NA",
    "RC_NA",
]

CATEGORICAL_VARS = [
    "sex",
    "incm",
    "edu",
    "educ",
    "sm_presnt",
    "dr_month",
    "pa_aerobic",
]

def pick_weight_col(df):
    for col in WEIGHT_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"No weight column found. Tried: {WEIGHT_CANDIDATES}")

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def weighted_mean(x, w):
    mask = x.notna() & w.notna() & (w > 0)
    x = x[mask]
    w = w[mask]
    if len(x) == 0:
        return np.nan
    return np.sum(w * x) / np.sum(w)

def weighted_sd(x, w):
    mask = x.notna() & w.notna() & (w > 0)
    x = x[mask]
    w = w[mask]
    if len(x) <= 1:
        return np.nan
    mu = weighted_mean(x, w)
    return np.sqrt(np.sum(w * (x - mu) ** 2) / np.sum(w))

def weighted_prop(y, w):
    mask = y.notna() & w.notna() & (w > 0)
    y = y[mask]
    w = w[mask]
    if len(y) == 0:
        return np.nan
    return np.sum(w * y) / np.sum(w)

def summarize_overall(df, weight_col):
    w = df[weight_col]
    rows = []

    for var in CONTINUOUS_VARS:
        if var not in df.columns:
            continue

        x = pd.to_numeric(df[var], errors="coerce")
        rows.append({
            "variable": var,
            "type": "continuous",
            "n_unweighted": x.notna().sum(),
            "weighted_mean": weighted_mean(x, w),
            "weighted_sd": weighted_sd(x, w),
        })

    for var in CATEGORICAL_VARS:
        if var not in df.columns:
            continue

        s = df[var]
        for level in sorted(s.dropna().unique()):
            y = (s == level).astype(float)
            rows.append({
                "variable": var,
                "type": "categorical",
                "level": level,
                "n_unweighted": s.notna().sum(),
                "weighted_percent": weighted_prop(y, w) * 100,
            })

    return pd.DataFrame(rows)

def summarize_targets(df, weight_col):
    w = df[weight_col]
    rows = []

    for target_name, label_col in TARGETS.items():
        if label_col not in df.columns:
            continue

        y = pd.to_numeric(df[label_col], errors="coerce")
        y = y.where(y.isin([0, 1]), np.nan)

        rows.append({
            "target": target_name,
            "label_col": label_col,
            "n_unweighted": y.notna().sum(),
            "n_positive_unweighted": int((y == 1).sum()),
            "prevalence_unweighted": y.mean(),
            "prevalence_weighted": weighted_prop(y, w),
            "prevalence_weighted_percent": weighted_prop(y, w) * 100,
        })

    return pd.DataFrame(rows)

def summarize_targets_by_sex(df, weight_col):
    rows = []

    if "sex" not in df.columns:
        return pd.DataFrame()

    for sex_value, sex_label in [(1, "male"), (2, "female")]:
        sub = df[df["sex"] == sex_value]
        w = sub[weight_col]

        for target_name, label_col in TARGETS.items():
            if label_col not in sub.columns:
                continue

            y = pd.to_numeric(sub[label_col], errors="coerce")
            y = y.where(y.isin([0, 1]), np.nan)

            rows.append({
                "sex": sex_label,
                "target": target_name,
                "label_col": label_col,
                "n_unweighted": y.notna().sum(),
                "n_positive_unweighted": int((y == 1).sum()),
                "prevalence_unweighted": y.mean(),
                "prevalence_weighted": weighted_prop(y, w),
                "prevalence_weighted_percent": weighted_prop(y, w) * 100,
            })

    return pd.DataFrame(rows)

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    weight_col = pick_weight_col(df)
    print(f"[WEIGHT COLUMN] {weight_col}")

    numeric_cols = (
        CONTINUOUS_VARS
        + CATEGORICAL_VARS
        + list(TARGETS.values())
        + [weight_col]
    )
    df = to_num(df, numeric_cols)

    df = df[df[weight_col].notna() & (df[weight_col] > 0)].copy()

    print(f"[INPUT SHAPE AFTER VALID WEIGHT] {df.shape}")

    overall = summarize_overall(df, weight_col)
    target = summarize_targets(df, weight_col)
    by_sex = summarize_targets_by_sex(df, weight_col)

    overall.to_csv(OUT_OVERALL, index=False, encoding="utf-8-sig")
    target.to_csv(OUT_TARGET, index=False, encoding="utf-8-sig")
    by_sex.to_csv(OUT_BY_SEX, index=False, encoding="utf-8-sig")

    print("\n[WEIGHTED TARGET PREVALENCE]")
    print(target)

    print("\n[SAVED]")
    print(OUT_OVERALL)
    print(OUT_TARGET)
    print(OUT_BY_SEX)

if __name__ == "__main__":
    main()