# ============================================
# 30_define_semihealthy_foodpattern_groups.py
# ============================================

import numpy as np
import pandas as pd
from pathlib import Path

from config import PROCESSED_DIR, RESULT_DIR

# --------------------------------------------
# PATHS
# --------------------------------------------

INPUT = PROCESSED_DIR / "analysis_cohort_energy_adjusted.csv"
FG_INPUT = PROCESSED_DIR / "ffq_food_groups.csv"

OUT_DATA = PROCESSED_DIR / "analysis_cohort_semihealthy_foodgroups.csv"
OUT_SUMMARY = RESULT_DIR / "30_semihealthy_group_summary.csv"

# --------------------------------------------
# TARGET / MEDICATION VARIABLES
# --------------------------------------------

DIAG_MED_COLS = [
    "HE_DMdg", "HE_DMdr",
    "HE_HPdg", "HE_HPdr",
    "HE_HLdg", "HE_HLdr",
]

FOOD_GROUP_Z = [
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

# --------------------------------------------
# UTIL
# --------------------------------------------

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def has_any_diagnosis_or_medication(df):
    """
    KNHANES disease diagnosis/medication variables are usually coded:
    1 = yes
    0/2/8/9 or missing = no/unknown depending variable.
    
    Here we conservatively treat 1 as diagnosed/medicated.
    """
    existing = [c for c in DIAG_MED_COLS if c in df.columns]
    if len(existing) == 0:
        raise ValueError("No diagnosis/medication columns found.")

    tmp = df[existing].apply(pd.to_numeric, errors="coerce")
    return tmp.eq(1).any(axis=1)

# --------------------------------------------
# GROUP DEFINITIONS
# --------------------------------------------

def define_glucose_group(df):
    """
    Diagnosis/medication-free only.

    normal_glucose:
        fasting glucose < 100 and HbA1c < 5.7
    prediabetes_clean:
        100 <= glucose < 126 or 5.7 <= HbA1c < 6.5
        with no DM diagnosis/medication
    """
    glu = pd.to_numeric(df["HE_glu"], errors="coerce")
    hba1c = pd.to_numeric(df["HE_HbA1c"], errors="coerce")

    normal = (glu < 100) & (hba1c < 5.7)
    pre = (
        ((glu >= 100) & (glu < 126)) |
        ((hba1c >= 5.7) & (hba1c < 6.5))
    )

    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[normal] = "normal_glucose"
    out[pre] = "prediabetes_clean"
    return out

def define_bp_group(df):
    """
    Diagnosis/medication-free only.

    normal_bp:
        SBP < 120 and DBP < 80
    prehypertension_clean:
        120 <= SBP < 140 or 80 <= DBP < 90
    """
    sbp = pd.to_numeric(df["HE_sbp"], errors="coerce")
    dbp = pd.to_numeric(df["HE_dbp"], errors="coerce")

    normal = (sbp < 120) & (dbp < 80)
    pre = (
        ((sbp >= 120) & (sbp < 140)) |
        ((dbp >= 80) & (dbp < 90))
    )

    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[normal] = "normal_bp"
    out[pre] = "prehypertension_clean"
    return out

def define_lipid_group(df):
    """
    Diagnosis/medication-free only.

    This is a practical borderline lipid-risk definition.
    Because exact borderline criteria can vary, we define:

    normal_lipid:
        TG < 150 and HDL >= 40(male)/50(female) and total cholesterol < 200

    borderline_lipid_clean:
        TG 150-199 or total cholesterol 200-239 or low HDL
        without lipid diagnosis/medication
    """
    tg = pd.to_numeric(df["HE_TG"], errors="coerce")
    chol = pd.to_numeric(df["HE_chol"], errors="coerce")
    hdl = pd.to_numeric(df["HE_HDL_st2"], errors="coerce")
    sex = pd.to_numeric(df["sex"], errors="coerce")

    low_hdl = ((sex == 1) & (hdl < 40)) | ((sex == 2) & (hdl < 50))

    normal = (
        (tg < 150) &
        (chol < 200) &
        (~low_hdl)
    )

    borderline = (
        ((tg >= 150) & (tg < 200)) |
        ((chol >= 200) & (chol < 240)) |
        low_hdl
    )

    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[normal] = "normal_lipid"
    out[borderline] = "borderline_lipid_clean"
    return out

def define_bmi_group(df):
    """
    Asian BMI cutoffs:
    normal_or_low: BMI < 23
    overweight_clean: 23 <= BMI < 25
    obesity: BMI >= 25
    """
    bmi = pd.to_numeric(df["HE_BMI"], errors="coerce")

    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[bmi < 23] = "normal_weight"
    out[(bmi >= 23) & (bmi < 25)] = "overweight_clean"
    out[bmi >= 25] = "obesity"
    return out

def define_any_semihealthy(df):
    """
    Diagnosis/medication-free integrated group.

    metabolically_normal:
        normal glucose + normal BP + normal lipid + BMI < 23

    semihealthy_any:
        at least one of:
        prediabetes_clean
        prehypertension_clean
        borderline_lipid_clean
        overweight_clean
        while remaining diagnosis/medication-free
    """
    g_glu = df["glucose_group"]
    g_bp = df["bp_group"]
    g_lipid = df["lipid_group"]
    g_bmi = df["bmi_group"]

    normal = (
        (g_glu == "normal_glucose") &
        (g_bp == "normal_bp") &
        (g_lipid == "normal_lipid") &
        (g_bmi == "normal_weight")
    )

    semi = (
        (g_glu == "prediabetes_clean") |
        (g_bp == "prehypertension_clean") |
        (g_lipid == "borderline_lipid_clean") |
        (g_bmi == "overweight_clean")
    )

    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[normal] = "metabolically_normal"
    out[semi & ~normal] = "semihealthy_any"
    return out

def make_binary_labels(df):
    """
    Binary labels:
    1 = semihealthy/pre-risk
    0 = corresponding normal reference
    """
    df["label_semihealthy_any"] = np.where(
        df["semihealthy_group"] == "semihealthy_any", 1,
        np.where(df["semihealthy_group"] == "metabolically_normal", 0, np.nan)
    )

    df["label_prediabetes_clean2"] = np.where(
        df["glucose_group"] == "prediabetes_clean", 1,
        np.where(df["glucose_group"] == "normal_glucose", 0, np.nan)
    )

    df["label_prehypertension_clean2"] = np.where(
        df["bp_group"] == "prehypertension_clean", 1,
        np.where(df["bp_group"] == "normal_bp", 0, np.nan)
    )

    df["label_borderline_lipid_clean"] = np.where(
        df["lipid_group"] == "borderline_lipid_clean", 1,
        np.where(df["lipid_group"] == "normal_lipid", 0, np.nan)
    )

    df["label_overweight_clean"] = np.where(
        df["bmi_group"] == "overweight_clean", 1,
        np.where(df["bmi_group"] == "normal_weight", 0, np.nan)
    )

    return df

def summarize_label(df, label_col):
    y = pd.to_numeric(df[label_col], errors="coerce")
    y = y[y.isin([0, 1])]
    return {
        "label": label_col,
        "n": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "positive_rate": y.mean() if len(y) > 0 else np.nan,
    }

# --------------------------------------------
# MAIN
# --------------------------------------------

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)
    fg = pd.read_csv(FG_INPUT, encoding="utf-8-sig", low_memory=False)

    print("[INPUT]")
    print(df.shape)
    print("[FOOD GROUP]")
    print(fg.shape)

    # merge by ID
    if "ID" not in df.columns or "ID" not in fg.columns:
        raise ValueError("ID column not found in either main dataset or food-group dataset.")

    df = df.merge(fg, on="ID", how="left")

    print("[MERGED]")
    print(df.shape)

    # numeric conversion
    numeric_cols = [
        "sex", "age", "HE_glu", "HE_HbA1c", "HE_sbp", "HE_dbp",
        "HE_TG", "HE_chol", "HE_HDL_st2", "HE_BMI",
    ] + DIAG_MED_COLS + FOOD_GROUP_Z

    df = to_num(df, numeric_cols)

    # exclude diagnosis/medication
    df["any_diagnosis_medication"] = has_any_diagnosis_or_medication(df)
    df_clean = df[df["any_diagnosis_medication"] == False].copy()

    print("[DIAGNOSIS/MEDICATION-FREE]")
    print(df_clean.shape)

    # define groups
    df_clean["glucose_group"] = define_glucose_group(df_clean)
    df_clean["bp_group"] = define_bp_group(df_clean)
    df_clean["lipid_group"] = define_lipid_group(df_clean)
    df_clean["bmi_group"] = define_bmi_group(df_clean)
    df_clean["semihealthy_group"] = define_any_semihealthy(df_clean)

    df_clean = make_binary_labels(df_clean)

    # save
    df_clean.to_csv(OUT_DATA, index=False, encoding="utf-8-sig")

    label_cols = [
        "label_semihealthy_any",
        "label_prediabetes_clean2",
        "label_prehypertension_clean2",
        "label_borderline_lipid_clean",
        "label_overweight_clean",
    ]

    summary = pd.DataFrame([summarize_label(df_clean, c) for c in label_cols])
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n[GROUP COUNTS]")
    for col in ["glucose_group", "bp_group", "lipid_group", "bmi_group", "semihealthy_group"]:
        print(f"\n{col}")
        print(df_clean[col].value_counts(dropna=False))

    print("\n[LABEL SUMMARY]")
    print(summary)

    print("\n[SAVED]")
    print(OUT_DATA)
    print(OUT_SUMMARY)

if __name__ == "__main__":
    main()