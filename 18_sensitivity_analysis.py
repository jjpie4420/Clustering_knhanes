import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import PROCESSED_DIR, RESULT_DIR

warnings.filterwarnings("ignore")

INPUT = PROCESSED_DIR / "analysis_cohort_energy_adjusted.csv"
OUTPUT = RESULT_DIR / "18_sensitivity_analysis.csv"

RANDOM_STATE = 42
N_SPLITS = 5

TARGETS = {
    "diabetes": {
        "label": "label_diabetes",
        "diag": "HE_DMdg",
        "med": "HE_DMdr",
    },
    "hypertension": {
        "label": "label_hypertension",
        "diag": "HE_HPdg",
        "med": "HE_HPdr",
    },
    "dyslipidemia": {
        "label": "label_dyslipidemia",
        "diag": "HE_HLdg",
        "med": "HE_HLdr",
    },
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

def get_auc(X, y):
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())

    if len(y) < 300 or n_pos < 30 or n_neg < 30:
        return np.nan

    n_splits = min(N_SPLITS, n_pos, n_neg)

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

    return roc_auc_score(y, pred)

###########################################################
# SENSITIVITY FILTERS
###########################################################

def apply_sensitivity(df, analysis_name, target_info):

    df = df.copy()

    #######################################################
    # stricter energy filter
    #######################################################

    if analysis_name == "strict_energy":

        df = df[
            (df["FQ_EN"].between(800, 4000)) &
            (df["RC_EN"].between(500, 4000))
        ]

    #######################################################
    # remove diagnosed
    #######################################################

    elif analysis_name == "remove_diagnosed":

        diag_col = target_info["diag"]

        if diag_col in df.columns:
            diag = pd.to_numeric(df[diag_col], errors="coerce")

            df = df[
                (~diag.isin([1]))
                | (diag.isna())
            ]

    #######################################################
    # remove medication
    #######################################################

    elif analysis_name == "remove_medication":

        med_col = target_info["med"]

        if med_col in df.columns:
            med = pd.to_numeric(df[med_col], errors="coerce")

            df = df[
                (~med.isin([1]))
                | (med.isna())
            ]

    #######################################################
    # balanced sample
    #######################################################

    elif analysis_name == "balanced_sample":
        pass

    return df

def balance_dataset(X, y):

    tmp = X.copy()
    tmp["y"] = y.values

    case = tmp[tmp["y"] == 1]
    control = tmp[tmp["y"] == 0]

    n = min(len(case), len(control))

    case_s = case.sample(
        n=n,
        random_state=RANDOM_STATE
    )

    control_s = control.sample(
        n=n,
        random_state=RANDOM_STATE
    )

    out = pd.concat([case_s, control_s])

    y_out = out["y"]
    X_out = out.drop(columns=["y"])

    return X_out, y_out

###########################################################
# MAIN
###########################################################

def main():

    df = pd.read_csv(
        INPUT,
        encoding="utf-8-sig",
        low_memory=False
    )

    ffq_features = existing(FFQ_ADJ, df)
    rc_features = existing(RC_ADJ, df)

    analyses = [
        "original",
        "strict_energy",
        "remove_diagnosed",
        "remove_medication",
        "balanced_sample",
    ]

    rows = []

    for target_name, target_info in TARGETS.items():

        label_col = target_info["label"]

        for analysis_name in analyses:

            if analysis_name == "original":
                tmp = df.copy()
            else:
                tmp = apply_sensitivity(
                    df,
                    analysis_name,
                    target_info
                )

            ###################################################
            # FFQ
            ###################################################

            X_ffq, y = clean_xy(
                tmp,
                label_col,
                ffq_features
            )

            ###################################################
            # RC
            ###################################################

            X_rc, y_rc = clean_xy(
                tmp,
                label_col,
                rc_features
            )

            common_idx = y.index.intersection(y_rc.index)

            X_ffq = X_ffq.loc[common_idx]
            X_rc = X_rc.loc[common_idx]
            y = y.loc[common_idx]

            ###################################################
            # balanced sample
            ###################################################

            if analysis_name == "balanced_sample":

                X_ffq, y = balance_dataset(X_ffq, y)

                X_rc = X_rc.loc[X_ffq.index]

            ###################################################
            # metrics
            ###################################################

            auc_ffq = get_auc(X_ffq, y)
            auc_rc = get_auc(X_rc, y)

            row = {
                "target": target_name,
                "analysis": analysis_name,

                "n": len(y),
                "n_positive": int(y.sum()),
                "prevalence": y.mean(),

                "auroc_ffq": auc_ffq,
                "auroc_rc": auc_rc,
                "delta_ffq_minus_rc": auc_ffq - auc_rc,
            }

            rows.append(row)

            print(
                f"[{target_name}]"
                f" [{analysis_name}]"
                f" n={len(y)}"
                f" auc_ffq={auc_ffq:.3f}"
                f" auc_rc={auc_rc:.3f}"
                f" delta={auc_ffq - auc_rc:.3f}"
            )

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[SUMMARY]\n")
    print(result)

    print(f"\n[SAVED] {OUTPUT}")

if __name__ == "__main__":
    main()