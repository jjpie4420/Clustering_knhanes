import numpy as np
import pandas as pd
from config import PROCESSED_DIR, RESULT_DIR

INPUT = PROCESSED_DIR / "analysis_cohort_energy_adjusted.csv"

OUT_OVERALL = RESULT_DIR / "16_table1_overall.csv"
OUT_BY_TARGET = RESULT_DIR / "16_table1_by_target.csv"

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
    "HE_LDL_drct",
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

VAR_LABELS = {
    "age": "Age, years",
    "sex": "Sex",
    "HE_BMI": "BMI, kg/m²",
    "HE_wc": "Waist circumference",
    "HE_sbp": "Systolic BP",
    "HE_dbp": "Diastolic BP",
    "HE_glu": "Fasting glucose",
    "HE_HbA1c": "HbA1c",
    "HE_chol": "Total cholesterol",
    "HE_HDL_st2": "HDL cholesterol",
    "HE_TG": "Triglycerides",
    "HE_LDL_drct": "LDL cholesterol",
    "FQ_EN": "FFQ energy",
    "RC_EN": "Recall energy",
    "FQ_PROT": "FFQ protein",
    "RC_PROT": "Recall protein",
    "FQ_FAT": "FFQ fat",
    "RC_FAT": "Recall fat",
    "FQ_CHO": "FFQ carbohydrate",
    "RC_CHO": "Recall carbohydrate",
    "FQ_TDF": "FFQ fiber",
    "RC_TDF": "Recall fiber",
    "FQ_NA": "FFQ sodium",
    "RC_NA": "Recall sodium",
    "incm": "Individual income",
    "edu": "Education",
    "educ": "Education recoded",
    "sm_presnt": "Current smoking",
    "dr_month": "Monthly drinking",
    "pa_aerobic": "Aerobic physical activity",
}

def to_numeric_safe(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def summarize_continuous(df, var):
    x = pd.to_numeric(df[var], errors="coerce").dropna()
    if len(x) == 0:
        return {
            "n": 0,
            "mean_sd": "",
            "median_iqr": "",
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
        }

    return {
        "n": len(x),
        "mean_sd": f"{x.mean():.2f} ± {x.std():.2f}",
        "median_iqr": f"{x.median():.2f} ({x.quantile(0.25):.2f}, {x.quantile(0.75):.2f})",
        "mean": x.mean(),
        "sd": x.std(),
        "median": x.median(),
        "q1": x.quantile(0.25),
        "q3": x.quantile(0.75),
    }

def summarize_categorical(df, var):
    s = df[var].dropna()
    n_nonmissing = len(s)
    rows = []

    if n_nonmissing == 0:
        return rows

    counts = s.value_counts(dropna=False).sort_index()

    for level, count in counts.items():
        pct = count / n_nonmissing * 100
        rows.append({
            "level": level,
            "n": n_nonmissing,
            "count_pct": f"{int(count)} ({pct:.1f}%)",
            "count": int(count),
            "pct": pct,
        })

    return rows

def build_overall_table(df):
    rows = []

    for var in CONTINUOUS_VARS:
        if var not in df.columns:
            continue

        stat = summarize_continuous(df, var)

        rows.append({
            "variable": var,
            "variable_label": VAR_LABELS.get(var, var),
            "type": "continuous",
            "level": "",
            "n": stat["n"],
            "overall": stat["mean_sd"],
            "median_iqr": stat["median_iqr"],
            "mean": stat["mean"],
            "sd": stat["sd"],
            "median": stat["median"],
            "q1": stat["q1"],
            "q3": stat["q3"],
        })

    for var in CATEGORICAL_VARS:
        if var not in df.columns:
            continue

        for stat in summarize_categorical(df, var):
            rows.append({
                "variable": var,
                "variable_label": VAR_LABELS.get(var, var),
                "type": "categorical",
                "level": stat["level"],
                "n": stat["n"],
                "overall": stat["count_pct"],
                "median_iqr": "",
                "mean": np.nan,
                "sd": np.nan,
                "median": np.nan,
                "q1": np.nan,
                "q3": np.nan,
            })

    return pd.DataFrame(rows)

def build_target_table(df):
    rows = []

    for target_name, label_col in TARGETS.items():
        if label_col not in df.columns:
            continue

        tmp = df[df[label_col].isin([0, 1])].copy()
        tmp[label_col] = tmp[label_col].astype(int)

        for group_value, group_name in [(0, "control"), (1, "case")]:
            g = tmp[tmp[label_col] == group_value]

            for var in CONTINUOUS_VARS:
                if var not in g.columns:
                    continue

                stat = summarize_continuous(g, var)

                rows.append({
                    "target": target_name,
                    "label_col": label_col,
                    "group": group_name,
                    "variable": var,
                    "variable_label": VAR_LABELS.get(var, var),
                    "type": "continuous",
                    "level": "",
                    "n_group": len(g),
                    "n_variable": stat["n"],
                    "value": stat["mean_sd"],
                    "median_iqr": stat["median_iqr"],
                    "mean": stat["mean"],
                    "sd": stat["sd"],
                    "median": stat["median"],
                    "q1": stat["q1"],
                    "q3": stat["q3"],
                })

            for var in CATEGORICAL_VARS:
                if var not in g.columns:
                    continue

                for stat in summarize_categorical(g, var):
                    rows.append({
                        "target": target_name,
                        "label_col": label_col,
                        "group": group_name,
                        "variable": var,
                        "variable_label": VAR_LABELS.get(var, var),
                        "type": "categorical",
                        "level": stat["level"],
                        "n_group": len(g),
                        "n_variable": stat["n"],
                        "value": stat["count_pct"],
                        "median_iqr": "",
                        "mean": np.nan,
                        "sd": np.nan,
                        "median": np.nan,
                        "q1": np.nan,
                        "q3": np.nan,
                    })

    return pd.DataFrame(rows)

def build_wide_target_table(long_df):
    wide = (
        long_df
        .pivot_table(
            index=["target", "variable", "variable_label", "type", "level"],
            columns="group",
            values="value",
            aggfunc="first"
        )
        .reset_index()
    )

    if "control" not in wide.columns:
        wide["control"] = ""
    if "case" not in wide.columns:
        wide["case"] = ""

    wide = wide[
        ["target", "variable", "variable_label", "type", "level", "control", "case"]
    ]

    return wide

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    needed_numeric = CONTINUOUS_VARS + list(TARGETS.values())
    df = to_numeric_safe(df, needed_numeric)

    print("[INPUT SHAPE]")
    print(df.shape)

    overall = build_overall_table(df)
    overall.to_csv(OUT_OVERALL, index=False, encoding="utf-8-sig")

    target_long = build_target_table(df)
    target_wide = build_wide_target_table(target_long)

    target_long.to_csv(
        RESULT_DIR / "16_table1_by_target_long.csv",
        index=False,
        encoding="utf-8-sig"
    )

    target_wide.to_csv(OUT_BY_TARGET, index=False, encoding="utf-8-sig")

    print("[SAVED]")
    print(OUT_OVERALL)
    print(OUT_BY_TARGET)
    print(RESULT_DIR / "16_table1_by_target_long.csv")

    print("\n[OVERALL PREVIEW]")
    print(overall.head(20))

    print("\n[TARGET PREVIEW]")
    print(target_wide.head(30))

if __name__ == "__main__":
    main()