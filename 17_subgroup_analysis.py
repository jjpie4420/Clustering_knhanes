import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from config import PROCESSED_DIR, RESULT_DIR

warnings.filterwarnings("ignore")

INPUT = PROCESSED_DIR / "analysis_cohort_energy_adjusted.csv"
OUTPUT = RESULT_DIR / "17_subgroup_auc_ffq_vs_rc.csv"

RANDOM_STATE = 42
N_SPLITS = 5

TARGETS = {
    "diabetes": "label_diabetes",
    "prediabetes_clean": "label_prediabetes_clean",
    "hypertension": "label_hypertension",
    "prehypertension_clean": "label_prehypertension_clean",
    "dyslipidemia": "label_dyslipidemia",
    "obesity": "label_obesity",
}

FFQ_ADJ = [
    "FQ_PROT_adj", "FQ_FAT_adj", "FQ_SFA_adj", "FQ_MUFA_adj",
    "FQ_PUFA_adj", "FQ_N3_adj", "FQ_N6_adj", "FQ_CHOL_adj",
    "FQ_CHO_adj", "FQ_TDF_adj", "FQ_CA_adj", "FQ_PHOS_adj",
    "FQ_FE_adj", "FQ_NA_adj", "FQ_K_adj", "FQ_VA_adj",
    "FQ_CAROT_adj", "FQ_RETIN_adj", "FQ_B1_adj", "FQ_B2_adj",
    "FQ_NIAC_adj", "FQ_VITC_adj",
]

RC_ADJ = [
    "RC_PROT_adj", "RC_FAT_adj", "RC_SFA_adj", "RC_MUFA_adj",
    "RC_PUFA_adj", "RC_N3_adj", "RC_N6_adj", "RC_CHOL_adj",
    "RC_CHO_adj", "RC_TDF_adj", "RC_CA_adj", "RC_PHOS_adj",
    "RC_FE_adj", "RC_NA_adj", "RC_K_adj", "RC_VA_adj",
    "RC_CAROT_adj", "RC_RETIN_adj", "RC_B1_adj", "RC_B2_adj",
    "RC_NIAC_adj", "RC_VITC_adj",
]

def existing(cols, df):
    return [c for c in cols if c in df.columns]

def make_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
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

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

def clean_xy(df, label_col, features):
    tmp = df[[label_col] + features].copy()
    tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce")
    tmp = tmp[tmp[label_col].isin([0, 1])]

    for c in features:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    X = tmp[features]
    y = tmp[label_col].astype(int)
    return X, y

def oof_auc(X, y):
    if len(y) < 300 or y.sum() < 30 or (y == 0).sum() < 30:
        return np.nan, np.nan

    n_splits = min(N_SPLITS, int(y.sum()), int((y == 0).sum()))
    if n_splits < 3:
        return np.nan, np.nan

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pred = cross_val_predict(
        make_pipeline(),
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return roc_auc_score(y, pred), average_precision_score(y, pred)

def add_subgroup_columns(df):
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df["sex_group"] = df["sex"].map({
        1: "male",
        2: "female",
        1.0: "male",
        2.0: "female",
    })

    df["age_group"] = pd.cut(
        df["age"],
        bins=[19, 39, 49, 64],
        labels=["20-39", "40-49", "50-64"],
        include_lowest=True,
    )

    df["bmi_group"] = pd.cut(
        pd.to_numeric(df["HE_BMI"], errors="coerce"),
        bins=[0, 23, 25, 100],
        labels=["normal_or_low", "overweight", "obese"],
        include_lowest=True,
    )

    return df

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)
    df = add_subgroup_columns(df)

    ffq_features = existing(FFQ_ADJ, df)
    rc_features = existing(RC_ADJ, df)

    subgroup_specs = [
        ("sex_group", "male"),
        ("sex_group", "female"),
        ("age_group", "20-39"),
        ("age_group", "40-49"),
        ("age_group", "50-64"),
        ("bmi_group", "normal_or_low"),
        ("bmi_group", "overweight"),
        ("bmi_group", "obese"),
    ]

    rows = []

    for subgroup_var, subgroup_value in subgroup_specs:
        sub = df[df[subgroup_var].astype(str) == str(subgroup_value)].copy()

        for target_name, label_col in TARGETS.items():
            if label_col not in sub.columns:
                continue

            X_ffq, y = clean_xy(sub, label_col, ffq_features)
            X_rc, y_rc = clean_xy(sub, label_col, rc_features)

            common_idx = y.index.intersection(y_rc.index)
            X_ffq = X_ffq.loc[common_idx]
            X_rc = X_rc.loc[common_idx]
            y = y.loc[common_idx]

            n = len(y)
            n_pos = int(y.sum())
            n_neg = int((y == 0).sum())

            if n < 300 or n_pos < 30 or n_neg < 30:
                print(f"[SKIP] {subgroup_var}={subgroup_value}, {target_name}: n={n}, pos={n_pos}")
                continue

            print(f"[RUN] {subgroup_var}={subgroup_value}, {target_name}, n={n}, pos={n_pos}")

            auc_ffq, auprc_ffq = oof_auc(X_ffq, y)
            auc_rc, auprc_rc = oof_auc(X_rc, y)

            rows.append({
                "subgroup_var": subgroup_var,
                "subgroup_value": subgroup_value,
                "target": target_name,
                "label_col": label_col,
                "n": n,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "prevalence": n_pos / n,
                "auroc_ffq": auc_ffq,
                "auroc_rc": auc_rc,
                "auroc_diff_ffq_minus_rc": auc_ffq - auc_rc,
                "auprc_ffq": auprc_ffq,
                "auprc_rc": auprc_rc,
                "auprc_diff_ffq_minus_rc": auprc_ffq - auprc_rc,
            })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("\n[SUMMARY]")
    print(result[[
        "subgroup_var", "subgroup_value", "target",
        "n", "prevalence", "auroc_ffq", "auroc_rc",
        "auroc_diff_ffq_minus_rc"
    ]])

    print(f"\n[SAVED] {OUTPUT}")

if __name__ == "__main__":
    main()