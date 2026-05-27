import numpy as np
import pandas as pd
from config import PROCESSED_DIR

INPUT = PROCESSED_DIR / "knhanes_ffq_rc_clean_labeled_2012_2016.csv"
OUTPUT = PROCESSED_DIR / "analysis_cohort_2012_2016.csv"

def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main():

    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    numeric_cols = [
        "age",
        "HE_BMI",
        "FQ_EN",
        "RC_EN",
    ]

    df = to_num(df, numeric_cols)

    print("\n[INITIAL]")
    print(df.shape)

    #################################################
    # 1. AGE FILTER
    #################################################

    df = df[(df["age"] >= 20) & (df["age"] <= 64)]

    print("\n[AFTER AGE FILTER]")
    print(df.shape)

    #################################################
    # 2. EXCLUDE PREGNANCY
    #################################################

    if "HE_prg" in df.columns:
        before = len(df)

        df = df[
            (df["HE_prg"].isna()) |
            (df["HE_prg"] != 1)
        ]

        print("\n[REMOVE PREGNANCY]")
        print(before, "->", len(df))

    #################################################
    # 3. KEEP FFQ AVAILABLE
    #################################################

    before = len(df)

    df = df[df["FQ_EN"].notna()]

    print("\n[KEEP FFQ]")
    print(before, "->", len(df))

    #################################################
    # 4. KEEP RC AVAILABLE
    #################################################

    before = len(df)

    df = df[df["RC_EN"].notna()]

    print("\n[KEEP RC]")
    print(before, "->", len(df))

    #################################################
    # 5. BMI AVAILABLE
    #################################################

    before = len(df)

    df = df[df["HE_BMI"].notna()]

    print("\n[KEEP BMI]")
    print(before, "->", len(df))

    #################################################
    # 6. EXTREME ENERGY FILTER
    #################################################

    # sex: 1=male, 2=female

    before = len(df)

    male_mask = (
        (df["sex"] == 1) &
        (
            (df["FQ_EN"] >= 800) &
            (df["FQ_EN"] <= 5000)
        )
    )

    female_mask = (
        (df["sex"] == 2) &
        (
            (df["FQ_EN"] >= 500) &
            (df["FQ_EN"] <= 4000)
        )
    )

    df = df[male_mask | female_mask]

    print("\n[EXTREME ENERGY FILTER]")
    print(before, "->", len(df))

    #################################################
    # 7. DUPLICATE REMOVE
    #################################################

    before = len(df)

    df = df.drop_duplicates(subset=["ID", "year"])

    print("\n[REMOVE DUPLICATES]")
    print(before, "->", len(df))

    #################################################
    # SAVE
    #################################################

    print("\n[FINAL SHAPE]")
    print(df.shape)

    print("\n[FINAL SEX]")
    print(df["sex"].value_counts())

    print("\n[FINAL AGE]")
    print(df["age"].describe())

    label_cols = [c for c in df.columns if c.startswith("label_")]

    print("\n[FINAL LABEL RATE]")
    print(df[label_cols].mean(numeric_only=True))

    df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n[SAVED] {OUTPUT}")

if __name__ == "__main__":
    main()