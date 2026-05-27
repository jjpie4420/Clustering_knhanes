import numpy as np
import pandas as pd
from config import PROCESSED_DIR

INPUT = PROCESSED_DIR / "knhanes_ffq_rc_merged_2012_2016.csv"
OUTPUT = PROCESSED_DIR / "knhanes_ffq_rc_labeled_2012_2016.csv"

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def define_labels(df):
    df = df.copy()

    for col in [
        "age", "HE_BMI", "HE_wc", "HE_sbp", "HE_dbp",
        "HE_glu", "HE_HbA1c", "HE_TG", "HE_HDL_st2",
        "HE_chol", "HE_LDL_drct"
    ]:
        if col in df.columns:
            df[col] = to_num(df[col])

    # Diabetes
    df["label_diabetes"] = np.where(
        (df["HE_glu"] >= 126) | (df["HE_HbA1c"] >= 6.5) | (df.get("HE_DM", np.nan) == 3),
        1,
        np.where(
            (df["HE_glu"].notna()) | (df["HE_HbA1c"].notna()),
            0,
            np.nan
        )
    )

    df["label_prediabetes"] = np.where(
        (
            ((df["HE_glu"] >= 100) & (df["HE_glu"] < 126)) |
            ((df["HE_HbA1c"] >= 5.7) & (df["HE_HbA1c"] < 6.5))
        ),
        1,
        np.where(
            (df["HE_glu"].notna()) | (df["HE_HbA1c"].notna()),
            0,
            np.nan
        )
    )

    # Hypertension
    df["label_hypertension"] = np.where(
        (df["HE_sbp"] >= 140) | (df["HE_dbp"] >= 90) | (df.get("HE_HP", np.nan) == 3),
        1,
        np.where(
            (df["HE_sbp"].notna()) | (df["HE_dbp"].notna()),
            0,
            np.nan
        )
    )

    df["label_prehypertension"] = np.where(
        (
            ((df["HE_sbp"] >= 120) & (df["HE_sbp"] < 140)) |
            ((df["HE_dbp"] >= 80) & (df["HE_dbp"] < 90))
        ),
        1,
        np.where(
            (df["HE_sbp"].notna()) | (df["HE_dbp"].notna()),
            0,
            np.nan
        )
    )

    # Obesity
    df["label_obesity"] = np.where(
        df["HE_BMI"] >= 25,
        1,
        np.where(df["HE_BMI"].notna(), 0, np.nan)
    )

    # Dyslipidemia proxy
    df["label_dyslipidemia"] = np.where(
        (df["HE_TG"] >= 200) |
        (df["HE_chol"] >= 240) |
        (df["HE_LDL_drct"] >= 160) |
        (df["HE_HDL_st2"] < 40),
        1,
        np.where(
            df[["HE_TG", "HE_chol", "HE_LDL_drct", "HE_HDL_st2"]].notna().any(axis=1),
            0,
            np.nan
        )
    )

    # Semi-healthy metabolic risk
    df["label_any_predisease"] = np.where(
        (df["label_prediabetes"] == 1) |
        (df["label_prehypertension"] == 1) |
        ((df["HE_BMI"] >= 23) & (df["HE_BMI"] < 25)),
        1,
        0
    )

    return df

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)
    df = define_labels(df)

    label_cols = [c for c in df.columns if c.startswith("label_")]
    print(df[label_cols].mean(numeric_only=True))
    print(df[label_cols].notna().sum())

    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[SAVED] {OUTPUT}")

if __name__ == "__main__":
    main()