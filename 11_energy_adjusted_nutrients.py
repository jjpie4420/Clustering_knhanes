import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import PROCESSED_DIR, RESULT_DIR

warnings.filterwarnings("ignore")

INPUT = PROCESSED_DIR / "analysis_cohort_2012_2016.csv"
OUTPUT = PROCESSED_DIR / "analysis_cohort_energy_adjusted.csv"

###########################################################
# TARGET NUTRIENTS
###########################################################

FFQ_NUTRIENTS = [
    "FQ_PROT",
    "FQ_FAT",
    "FQ_SFA",
    "FQ_MUFA",
    "FQ_PUFA",
    "FQ_N3",
    "FQ_N6",
    "FQ_CHOL",
    "FQ_CHO",
    "FQ_TDF",
    "FQ_CA",
    "FQ_PHOS",
    "FQ_FE",
    "FQ_NA",
    "FQ_K",
    "FQ_VA",
    "FQ_CAROT",
    "FQ_RETIN",
    "FQ_B1",
    "FQ_B2",
    "FQ_NIAC",
    "FQ_VITC",
]

RC_NUTRIENTS = [
    "RC_PROT",
    "RC_FAT",
    "RC_SFA",
    "RC_MUFA",
    "RC_PUFA",
    "RC_N3",
    "RC_N6",
    "RC_CHOL",
    "RC_CHO",
    "RC_TDF",
    "RC_CA",
    "RC_PHOS",
    "RC_FE",
    "RC_NA",
    "RC_K",
    "RC_VA",
    "RC_CAROT",
    "RC_RETIN",
    "RC_B1",
    "RC_B2",
    "RC_NIAC",
    "RC_VITC",
]

###########################################################
# FUNCTIONS
###########################################################

def residual_adjust(df, nutrient_col, energy_col):

    tmp = df[[nutrient_col, energy_col]].copy()

    tmp[nutrient_col] = pd.to_numeric(
        tmp[nutrient_col],
        errors="coerce"
    )

    tmp[energy_col] = pd.to_numeric(
        tmp[energy_col],
        errors="coerce"
    )

    tmp = tmp.dropna()

    if len(tmp) < 100:
        return pd.Series(np.nan, index=df.index)

    X = tmp[[energy_col]]
    y = tmp[nutrient_col]

    model = LinearRegression()
    model.fit(X, y)

    predicted = model.predict(X)

    residual = y - predicted

    adjusted = residual + y.mean()

    out = pd.Series(np.nan, index=df.index)

    out.loc[tmp.index] = adjusted

    return out

###########################################################
# MAIN
###########################################################

def main():

    df = pd.read_csv(
        INPUT,
        encoding="utf-8-sig",
        low_memory=False
    )

    ###########################################################
    # FFQ residual adjustment
    ###########################################################

    print("\n[FFQ ENERGY ADJUSTMENT]\n")

    for nutrient in FFQ_NUTRIENTS:

        if nutrient not in df.columns:
            continue

        adj_col = nutrient + "_adj"

        print(f"Processing: {adj_col}")

        df[adj_col] = residual_adjust(
            df,
            nutrient,
            "FQ_EN"
        )

    ###########################################################
    # RC residual adjustment
    ###########################################################

    print("\n[RC ENERGY ADJUSTMENT]\n")

    for nutrient in RC_NUTRIENTS:

        if nutrient not in df.columns:
            continue

        adj_col = nutrient + "_adj"

        print(f"Processing: {adj_col}")

        df[adj_col] = residual_adjust(
            df,
            nutrient,
            "RC_EN"
        )

    ###########################################################
    # SUMMARY
    ###########################################################

    adj_cols = [
        c for c in df.columns
        if c.endswith("_adj")
    ]

    print("\n[ADJUSTED VARIABLES]\n")
    print(adj_cols)

    print(f"\nTotal adjusted variables: {len(adj_cols)}")

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