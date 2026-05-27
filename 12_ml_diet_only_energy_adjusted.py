import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import PROCESSED_DIR, RESULT_DIR

warnings.filterwarnings("ignore")

INPUT = PROCESSED_DIR / "analysis_cohort_energy_adjusted.csv"
OUTPUT = RESULT_DIR / "12_ml_energy_adjusted_results.csv"

RANDOM_STATE = 42
N_SPLITS = 5

###########################################################
# TARGETS
###########################################################

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

###########################################################
# ADJUSTED FEATURES
###########################################################

FFQ_ADJ = [
    "FQ_PROT_adj",
    "FQ_FAT_adj",
    "FQ_SFA_adj",
    "FQ_MUFA_adj",
    "FQ_PUFA_adj",
    "FQ_N3_adj",
    "FQ_N6_adj",
    "FQ_CHOL_adj",
    "FQ_CHO_adj",
    "FQ_TDF_adj",
    "FQ_CA_adj",
    "FQ_PHOS_adj",
    "FQ_FE_adj",
    "FQ_NA_adj",
    "FQ_K_adj",
    "FQ_VA_adj",
    "FQ_CAROT_adj",
    "FQ_RETIN_adj",
    "FQ_B1_adj",
    "FQ_B2_adj",
    "FQ_NIAC_adj",
    "FQ_VITC_adj",
]

RC_ADJ = [
    "RC_PROT_adj",
    "RC_FAT_adj",
    "RC_SFA_adj",
    "RC_MUFA_adj",
    "RC_PUFA_adj",
    "RC_N3_adj",
    "RC_N6_adj",
    "RC_CHOL_adj",
    "RC_CHO_adj",
    "RC_TDF_adj",
    "RC_CA_adj",
    "RC_PHOS_adj",
    "RC_FE_adj",
    "RC_NA_adj",
    "RC_K_adj",
    "RC_VA_adj",
    "RC_CAROT_adj",
    "RC_RETIN_adj",
    "RC_B1_adj",
    "RC_B2_adj",
    "RC_NIAC_adj",
    "RC_VITC_adj",
]

###########################################################
# MODELS
###########################################################

MODELS = {
    "logistic_l2": LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}

###########################################################
# FUNCTIONS
###########################################################

def existing(cols, df):
    return [c for c in cols if c in df.columns]


def build_feature_sets(df):

    ffq = existing(FFQ_ADJ, df)
    rc = existing(RC_ADJ, df)

    return {
        "ffq_adj_only": ffq,
        "rc_adj_only": rc,
        "ffq_rc_adj_combined": ffq + rc,
    }


def clean_xy(df, label_col, features):

    tmp = df[[label_col] + features].copy()

    tmp[label_col] = pd.to_numeric(
        tmp[label_col],
        errors="coerce"
    )

    tmp = tmp[tmp[label_col].isin([0, 1])]

    for c in features:
        tmp[c] = pd.to_numeric(
            tmp[c],
            errors="coerce"
        )

    X = tmp[features]
    y = tmp[label_col].astype(int)

    return X, y


def make_pipeline(model):

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, slice(0, None)),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipe


def evaluate_cv(X, y, model):

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pipe = make_pipeline(model)

    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
    }

    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    out = {}

    for metric in scoring.keys():

        vals = scores[f"test_{metric}"]

        out[f"{metric}_mean"] = np.mean(vals)
        out[f"{metric}_sd"] = np.std(vals)

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

    feature_sets = build_feature_sets(df)

    rows = []

    for target_name, label_col in TARGETS.items():

        if label_col not in df.columns:
            continue

        for feature_set_name, features in feature_sets.items():

            if len(features) == 0:
                continue

            X, y = clean_xy(
                df,
                label_col,
                features
            )

            n = len(y)
            n_pos = int(y.sum())
            prevalence = n_pos / n

            if n < 200:
                continue

            print(
                f"\n[TARGET] {target_name}"
                f" | [SET] {feature_set_name}"
                f" | n={n}"
                f" | pos={n_pos}"
                f" | prev={prevalence:.3f}"
                f" | p={len(features)}"
            )

            for model_name, model in MODELS.items():

                print(f"  - running {model_name}")

                result = evaluate_cv(
                    X,
                    y,
                    model
                )

                row = {
                    "target": target_name,
                    "label_col": label_col,
                    "feature_set": feature_set_name,
                    "model": model_name,

                    "n": n,
                    "n_positive": n_pos,
                    "prevalence": prevalence,

                    "n_features": len(features),
                }

                row.update(result)

                rows.append(row)

    result_df = pd.DataFrame(rows)

    result_df.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[SUMMARY]\n")

    show_cols = [
        "target",
        "feature_set",
        "model",
        "roc_auc_mean",
        "average_precision_mean",
        "balanced_accuracy_mean",
        "f1_mean",
    ]

    print(
        result_df[show_cols]
        .sort_values(
            ["target", "roc_auc_mean"],
            ascending=[True, False]
        )
    )

    print(f"\n[SAVED] {OUTPUT}")

if __name__ == "__main__":
    main()