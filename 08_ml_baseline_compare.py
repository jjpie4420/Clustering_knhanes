import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from config import PROCESSED_DIR, RESULT_DIR

warnings.filterwarnings("ignore")

INPUT = PROCESSED_DIR / "analysis_cohort_2012_2016.csv"
OUTPUT = RESULT_DIR / "08_ml_baseline_compare_results.csv"

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

CLINICAL_BASE = [
    "age",
    "sex",
    "incm",
    "edu",
    "educ",
    "sm_presnt",
    "dr_month",
    "pa_aerobic",
    "HE_BMI",
]

TARGET_EXCLUDE = {
    "obesity": ["HE_BMI"],
}

NUTRIENT_PAIRS = {
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
        max_depth=None,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}


def existing(cols, df):
    return [c for c in cols if c in df.columns]


def get_feature_sets(df, target_name):
    clinical = existing(CLINICAL_BASE, df)

    exclude = TARGET_EXCLUDE.get(target_name, [])
    clinical = [c for c in clinical if c not in exclude]

    ffq_nutrients = []
    rc_nutrients = []

    for _, (fq, rc) in NUTRIENT_PAIRS.items():
        if fq in df.columns and rc in df.columns:
            ffq_nutrients.append(fq)
            rc_nutrients.append(rc)

    return {
        "clinical": clinical,
        "clinical_ffq_nutrient": clinical + ffq_nutrients,
        "clinical_rc_nutrient": clinical + rc_nutrients,
        "clinical_ffq_rc_nutrient": clinical + ffq_nutrients + rc_nutrients,
    }


def clean_xy(df, label_col, features):
    use_cols = [label_col] + features
    tmp = df[use_cols].copy()

    tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce")
    tmp = tmp[tmp[label_col].isin([0, 1])]

    for c in features:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    y = tmp[label_col].astype(int)
    X = tmp[features]

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


def evaluate_cv(X, y, model_name, model):
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


def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    rows = []

    for target_name, label_col in TARGETS.items():
        if label_col not in df.columns:
            print(f"[SKIP] {target_name}: {label_col} not found")
            continue

        feature_sets = get_feature_sets(df, target_name)

        for feature_set_name, features in feature_sets.items():
            features = existing(features, df)

            if len(features) == 0:
                print(f"[SKIP] {target_name} / {feature_set_name}: no features")
                continue

            X, y = clean_xy(df, label_col, features)

            n = len(y)
            n_pos = int(y.sum())
            n_neg = int((y == 0).sum())
            prevalence = n_pos / n if n > 0 else np.nan

            if n < 200 or n_pos < 50 or n_neg < 50:
                print(f"[SKIP] {target_name} / {feature_set_name}: insufficient n")
                continue

            print(
                f"\n[TARGET] {target_name} | [SET] {feature_set_name} "
                f"| n={n}, pos={n_pos}, prev={prevalence:.3f}, p={len(features)}"
            )

            for model_name, model in MODELS.items():
                print(f"  - running {model_name}")

                result = evaluate_cv(X, y, model_name, model)

                row = {
                    "target": target_name,
                    "label_col": label_col,
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "n": n,
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                    "prevalence": prevalence,
                    "n_features": len(features),
                    "features": ",".join(features),
                }
                row.update(result)
                rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("\n[SUMMARY]")
    show_cols = [
        "target",
        "feature_set",
        "model",
        "n",
        "prevalence",
        "n_features",
        "roc_auc_mean",
        "average_precision_mean",
        "balanced_accuracy_mean",
        "f1_mean",
    ]
    print(result_df[show_cols].sort_values(["target", "model", "roc_auc_mean"]))

    print(f"\n[SAVED] {OUTPUT}")


if __name__ == "__main__":
    main()