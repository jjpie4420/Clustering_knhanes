import numpy as np
import pandas as pd
from config import PROCESSED_DIR

INPUT = PROCESSED_DIR / "knhanes_ffq_rc_merged_2012_2016.csv"
OUTPUT = PROCESSED_DIR / "knhanes_ffq_rc_clean_labeled_2012_2016.csv"


###########################################################
# UTILS
###########################################################

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def any_yes(df, cols, yes_values=[1, 2, 3]):
    """
    여러 diagnosis / medication 변수 중 하나라도 yes면 True
    """
    existing = [c for c in cols if c in df.columns]

    if len(existing) == 0:
        return pd.Series(False, index=df.index)

    out = pd.Series(False, index=df.index)

    for c in existing:
        out = out | df[c].isin(yes_values)

    return out


###########################################################
# MAIN
###########################################################

def main():

    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    ###########################################################
    # NUMERIC CONVERSION
    ###########################################################

    numeric_cols = [
        "HE_glu",
        "HE_HbA1c",
        "HE_sbp",
        "HE_dbp",
        "HE_BMI",
        "HE_TG",
        "HE_chol",
        "HE_HDL_st2",
        "HE_LDL_drct",
    ]

    df = to_num(df, numeric_cols)

    ###########################################################
    # DIAGNOSIS / MEDICATION FLAGS
    ###########################################################

    # Diabetes
    dm_diag = any_yes(df, [
        "HE_DMdg",
        "DE1_dg"
    ])

    dm_med = any_yes(df, [
        "HE_DMdr",
        "DE1_pt"
    ])

    # Hypertension
    htn_diag = any_yes(df, [
        "HE_HPdg",
        "DI1_dg"
    ])

    htn_med = any_yes(df, [
        "HE_HPdr",
        "DI1_pt"
    ])

    # Dyslipidemia
    dys_diag = any_yes(df, [
        "HE_HLdg",
        "DI2_dg"
    ])

    dys_med = any_yes(df, [
        "HE_HLdr",
        "DI2_pt"
    ])

    ###########################################################
    # DIABETES
    ###########################################################

    diabetes_lab = (
        (df["HE_glu"] >= 126) |
        (df["HE_HbA1c"] >= 6.5)
    )

    prediabetes_lab = (
        (
            (df["HE_glu"] >= 100) &
            (df["HE_glu"] < 126)
        ) |
        (
            (df["HE_HbA1c"] >= 5.7) &
            (df["HE_HbA1c"] < 6.5)
        )
    )

    ###########################################################
    # MAIN DIABETES
    ###########################################################

    df["label_diabetes"] = np.where(
        diabetes_lab | dm_diag | dm_med,
        1,
        np.where(
            df[["HE_glu", "HE_HbA1c"]].notna().any(axis=1),
            0,
            np.nan
        )
    )

    ###########################################################
    # UNTREATED DIABETES
    ###########################################################

    df["label_diabetes_untreated"] = np.where(
        diabetes_lab & (~dm_med),
        1,
        np.where(
            df[["HE_glu", "HE_HbA1c"]].notna().any(axis=1),
            0,
            np.nan
        )
    )

    ###########################################################
    # CLEAN PREDIABETES
    ###########################################################

    clean_predm_case = (
        prediabetes_lab &
        (~dm_diag) &
        (~dm_med) &
        (~diabetes_lab)
    )

    clean_predm_control = (
        (df["HE_glu"] < 100) &
        (df["HE_HbA1c"] < 5.7) &
        (~dm_diag) &
        (~dm_med)
    )

    df["label_prediabetes_clean"] = np.where(
        clean_predm_case,
        1,
        np.where(
            clean_predm_control,
            0,
            np.nan
        )
    )

    ###########################################################
    # HYPERTENSION
    ###########################################################

    htn_lab = (
        (df["HE_sbp"] >= 140) |
        (df["HE_dbp"] >= 90)
    )

    prehtn_lab = (
        (
            (df["HE_sbp"] >= 120) &
            (df["HE_sbp"] < 140)
        ) |
        (
            (df["HE_dbp"] >= 80) &
            (df["HE_dbp"] < 90)
        )
    )

    ###########################################################
    # MAIN HTN
    ###########################################################

    df["label_hypertension"] = np.where(
        htn_lab | htn_diag | htn_med,
        1,
        np.where(
            df[["HE_sbp", "HE_dbp"]].notna().any(axis=1),
            0,
            np.nan
        )
    )

    ###########################################################
    # UNTREATED HTN
    ###########################################################

    df["label_hypertension_untreated"] = np.where(
        htn_lab & (~htn_med),
        1,
        np.where(
            df[["HE_sbp", "HE_dbp"]].notna().any(axis=1),
            0,
            np.nan
        )
    )

    ###########################################################
    # CLEAN PREHTN
    ###########################################################

    clean_prehtn_case = (
        prehtn_lab &
        (~htn_diag) &
        (~htn_med) &
        (~htn_lab)
    )

    clean_prehtn_control = (
        (df["HE_sbp"] < 120) &
        (df["HE_dbp"] < 80) &
        (~htn_diag) &
        (~htn_med)
    )

    df["label_prehypertension_clean"] = np.where(
        clean_prehtn_case,
        1,
        np.where(
            clean_prehtn_control,
            0,
            np.nan
        )
    )

    ###########################################################
    # DYSLIPIDEMIA
    ###########################################################

    dys_lab = (
        (df["HE_TG"] >= 200) |
        (df["HE_chol"] >= 240) |
        (df["HE_HDL_st2"] < 40)
    )

    ###########################################################
    # MAIN DYSLIPIDEMIA
    ###########################################################

    df["label_dyslipidemia"] = np.where(
        dys_lab | dys_diag | dys_med,
        1,
        np.where(
            df[["HE_TG", "HE_chol", "HE_HDL_st2"]].notna().any(axis=1),
            0,
            np.nan
        )
    )

    ###########################################################
    # UNTREATED DYSLIPIDEMIA
    ###########################################################

    df["label_dyslipidemia_untreated"] = np.where(
        dys_lab & (~dys_med),
        1,
        np.where(
            df[["HE_TG", "HE_chol", "HE_HDL_st2"]].notna().any(axis=1),
            0,
            np.nan
        )
    )

    ###########################################################
    # OBESITY
    ###########################################################

    df["label_obesity"] = np.where(
        df["HE_BMI"] >= 25,
        1,
        np.where(
            df["HE_BMI"].notna(),
            0,
            np.nan
        )
    )

    ###########################################################
    # SUMMARY
    ###########################################################

    label_cols = [c for c in df.columns if c.startswith("label_")]

    print("\n[LABEL POSITIVE RATE]\n")
    print(df[label_cols].mean(numeric_only=True))

    print("\n[LABEL N]\n")
    print(df[label_cols].notna().sum())

    ###########################################################
    # SAVE
    ###########################################################

    df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n[SAVED] {OUTPUT}")


if __name__ == "__main__":
    main()