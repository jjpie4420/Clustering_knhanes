import pandas as pd
from config import PROCESSED_DIR, RESULT_DIR

INPUT = PROCESSED_DIR / "knhanes_ffq_rc_labeled_2012_2016.csv"

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    label_cols = [c for c in df.columns if c.startswith("label_")]
    ffq_cols = [c for c in df.columns if c.startswith(("FQ_", "FF_", "FA_", "FS_"))]
    rc_cols = [c for c in df.columns if c.startswith("RC_")]

    print("\n[SHAPE]")
    print(df.shape)

    print("\n[YEAR DISTRIBUTION]")
    print(df["year"].value_counts(dropna=False).sort_index())

    print("\n[SEX DISTRIBUTION]")
    print(df["sex"].value_counts(dropna=False).sort_index())

    print("\n[AGE SUMMARY]")
    print(pd.to_numeric(df["age"], errors="coerce").describe())

    print("\n[LABEL COUNTS]")
    label_summary = []
    for col in label_cols:
        vc = df[col].value_counts(dropna=False).to_dict()
        label_summary.append({
            "label": col,
            "n_total": len(df),
            "n_nonmissing": df[col].notna().sum(),
            "n_positive": (df[col] == 1).sum(),
            "n_negative": (df[col] == 0).sum(),
            "positive_rate": (df[col] == 1).mean()
        })
    label_summary = pd.DataFrame(label_summary)
    print(label_summary)

    print("\n[FEATURE BLOCK COUNTS]")
    print({
        "ffq_cols": len(ffq_cols),
        "rc_cols": len(rc_cols),
        "label_cols": len(label_cols)
    })

    key_vars = [
        "age", "sex", "HE_BMI", "HE_wc", "HE_sbp", "HE_dbp",
        "HE_glu", "HE_HbA1c", "HE_TG", "HE_HDL_st2",
        "HE_chol", "HE_LDL_drct",
        "FQ_EN", "FQ_PROT", "FQ_FAT", "FQ_CHO", "FQ_TDF",
        "RC_EN", "RC_PROT", "RC_FAT", "RC_CHO", "RC_TDF"
    ]
    key_vars = [c for c in key_vars if c in df.columns]

    missing = (
        df[key_vars]
        .isna()
        .mean()
        .reset_index()
        .rename(columns={"index": "variable", 0: "missing_rate"})
        .sort_values("missing_rate", ascending=False)
    )

    print("\n[KEY VARIABLE MISSING RATE]")
    print(missing)

    label_summary.to_csv(RESULT_DIR / "05_label_summary.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(RESULT_DIR / "05_key_missing_rate.csv", index=False, encoding="utf-8-sig")

    year_label = df.groupby("year")[label_cols].mean(numeric_only=True)
    year_label.to_csv(RESULT_DIR / "05_label_rate_by_year.csv", encoding="utf-8-sig")

    print("\n[SAVED]")
    print(RESULT_DIR / "05_label_summary.csv")
    print(RESULT_DIR / "05_key_missing_rate.csv")
    print(RESULT_DIR / "05_label_rate_by_year.csv")

if __name__ == "__main__":
    main()