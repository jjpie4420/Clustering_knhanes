import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from config import PROCESSED_DIR, RESULT_DIR

warnings.filterwarnings("ignore")

INPUT = PROCESSED_DIR / "analysis_cohort_2012_2016.csv"
OUTPUT = RESULT_DIR / "10_logistic_feature_importance.csv"

RANDOM_STATE = 42
N_SPLITS = 5

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

PAIR_NUTRIENTS = {
    "EN": ("FQ_EN", "RC_EN"),
    "PROT": ("FQ_PROT", "RC_PROT"),
    "FAT": ("FQ_FAT", "RC_FAT"),
    "SFA": ("FQ_SFA", "RC_SFA"),
    "MUFA": ("FQ_MUFA", "RC_MUFA"),
    "PUFA": ("FQ_PUFA", "RC_PUFA"),
    "N3": ("FQ_N3", "RC_N3"),
    "N6": ("FQ_N6", "RC_N6"),
    "CHOL": ("FQ_CHOL", "RC_CHOL"),
    "CHO": ("FQ_CHO", "RC_CHO"),
    "TDF": ("FQ_TDF", "RC_TDF"),
    "CA": ("FQ_CA", "RC_CA"),
    "PHOS": ("FQ_PHOS", "RC_PHOS"),
    "FE": ("FQ_FE", "RC_FE"),
    "NA": ("FQ_NA", "RC_NA"),
    "K": ("FQ_K", "RC_K"),
    "VA": ("FQ_VA", "RC_VA"),
    "CAROT": ("FQ_CAROT", "RC_CAROT"),
    "RETIN": ("FQ_RETIN", "RC_RETIN"),
    "B1": ("FQ_B1", "RC_B1"),
    "B2": ("FQ_B2", "RC_B2"),
    "NIAC": ("FQ_NIAC", "RC_NIAC"),
    "VITC": ("FQ_VITC", "RC_VITC"),
}

def build_feature_sets(df):
    ffq = []
    rc = []

    for _, (fq, rc_col) in PAIR_NUTRIENTS.items():
        if fq in df.columns and rc_col in df.columns:
            ffq.append(fq)
            rc.append(rc_col)

    return {
        "ffq_only": ffq,
        "rc_only": rc,
        "ffq_rc_combined": ffq + rc,
    }

def clean_xy(df, label_col, features):
    tmp = df[[label_col] + features].copy()

    tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce")
    tmp = tmp[tmp[label_col].isin([0, 1])]

    for c in features:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    X = tmp[features]
    y = tmp[label_col].astype(int)

    return X, y

def make_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                slice(0, None),
            )
        ]
    )

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

def extract_cv_coefficients(X, y, features):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    coef_list = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        pipe = make_pipeline()
        pipe.fit(X_train, y_train)

        coef = pipe.named_steps["model"].coef_[0]

        fold_df = pd.DataFrame({
            "fold": fold,
            "feature": features,
            "coef": coef,
            "abs_coef": np.abs(coef),
        })

        coef_list.append(fold_df)

    coef_df = pd.concat(coef_list, axis=0, ignore_index=True)

    summary = (
        coef_df
        .groupby("feature", as_index=False)
        .agg(
            coef_mean=("coef", "mean"),
            coef_sd=("coef", "std"),
            abs_coef_mean=("abs_coef", "mean"),
            abs_coef_sd=("abs_coef", "std"),
        )
    )

    summary["direction"] = np.where(
        summary["coef_mean"] > 0,
        "positive",
        "negative"
    )

    return summary.sort_values("abs_coef_mean", ascending=False)

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    feature_sets = build_feature_sets(df)
    all_rows = []

    for target_name, label_col in TARGETS.items():
        if label_col not in df.columns:
            continue

        for feature_set_name, features in feature_sets.items():
            if len(features) == 0:
                continue

            X, y = clean_xy(df, label_col, features)

            n = len(y)
            n_pos = int(y.sum())
            prevalence = n_pos / n

            if n < 200 or n_pos < 30:
                continue

            print(
                f"\n[TARGET] {target_name}"
                f" | [SET] {feature_set_name}"
                f" | n={n}"
                f" | pos={n_pos}"
                f" | prev={prevalence:.3f}"
                f" | p={len(features)}"
            )

            coef_summary = extract_cv_coefficients(X, y, features)

            coef_summary.insert(0, "target", target_name)
            coef_summary.insert(1, "label_col", label_col)
            coef_summary.insert(2, "feature_set", feature_set_name)
            coef_summary.insert(3, "n", n)
            coef_summary.insert(4, "n_positive", n_pos)
            coef_summary.insert(5, "prevalence", prevalence)

            all_rows.append(coef_summary)

            print(
                coef_summary[
                    ["feature", "coef_mean", "abs_coef_mean", "direction"]
                ].head(10)
            )

    result = pd.concat(all_rows, axis=0, ignore_index=True)

    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"\n[SAVED] {OUTPUT}")

if __name__ == "__main__":
    main()