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
OUTPUT = RESULT_DIR / "14_auc_bootstrap_ffq_vs_rc.csv"

RANDOM_STATE = 42
N_SPLITS = 5
N_BOOTSTRAP = 1000

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

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipe

def get_oof_predictions(X, y):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pipe = make_pipeline()

    pred = cross_val_predict(
        pipe,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return pred

def bootstrap_metric_diff(y, pred_ffq, pred_rc, metric_func):
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)

    diffs = []

    y_arr = np.asarray(y)
    ffq_arr = np.asarray(pred_ffq)
    rc_arr = np.asarray(pred_rc)

    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)

        y_b = y_arr[idx]

        if len(np.unique(y_b)) < 2:
            continue

        ffq_b = ffq_arr[idx]
        rc_b = rc_arr[idx]

        diff = metric_func(y_b, ffq_b) - metric_func(y_b, rc_b)
        diffs.append(diff)

    diffs = np.array(diffs)

    return {
        "diff_mean": np.mean(diffs),
        "diff_median": np.median(diffs),
        "diff_ci_lower": np.percentile(diffs, 2.5),
        "diff_ci_upper": np.percentile(diffs, 97.5),
        "p_bootstrap_two_sided": 2 * min(
            np.mean(diffs <= 0),
            np.mean(diffs >= 0)
        ),
        "n_bootstrap_valid": len(diffs),
    }

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    ffq_features = existing(FFQ_ADJ, df)
    rc_features = existing(RC_ADJ, df)

    rows = []

    for target_name, label_col in TARGETS.items():
        if label_col not in df.columns:
            continue

        X_ffq, y = clean_xy(df, label_col, ffq_features)
        X_rc, y_rc = clean_xy(df, label_col, rc_features)

        # safety check
        if not y.index.equals(y_rc.index):
            common_idx = y.index.intersection(y_rc.index)
            y = y.loc[common_idx]
            X_ffq = X_ffq.loc[common_idx]
            X_rc = X_rc.loc[common_idx]

        n = len(y)
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        prevalence = n_pos / n

        if n < 200 or n_pos < 30 or n_neg < 30:
            print(f"[SKIP] {target_name}: insufficient sample")
            continue

        print(
            f"\n[TARGET] {target_name}"
            f" | n={n}"
            f" | pos={n_pos}"
            f" | prev={prevalence:.3f}"
        )

        print("  - OOF prediction: FFQ")
        pred_ffq = get_oof_predictions(X_ffq, y)

        print("  - OOF prediction: RC")
        pred_rc = get_oof_predictions(X_rc, y)

        auc_ffq = roc_auc_score(y, pred_ffq)
        auc_rc = roc_auc_score(y, pred_rc)
        auprc_ffq = average_precision_score(y, pred_ffq)
        auprc_rc = average_precision_score(y, pred_rc)

        print(f"  - AUROC FFQ={auc_ffq:.4f}, RC={auc_rc:.4f}, diff={auc_ffq - auc_rc:.4f}")

        auc_diff = bootstrap_metric_diff(
            y=y,
            pred_ffq=pred_ffq,
            pred_rc=pred_rc,
            metric_func=roc_auc_score,
        )

        auprc_diff = bootstrap_metric_diff(
            y=y,
            pred_ffq=pred_ffq,
            pred_rc=pred_rc,
            metric_func=average_precision_score,
        )

        row = {
            "target": target_name,
            "label_col": label_col,
            "n": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "prevalence": prevalence,

            "auroc_ffq": auc_ffq,
            "auroc_rc": auc_rc,
            "auroc_diff_ffq_minus_rc": auc_ffq - auc_rc,
            "auroc_diff_boot_mean": auc_diff["diff_mean"],
            "auroc_diff_ci_lower": auc_diff["diff_ci_lower"],
            "auroc_diff_ci_upper": auc_diff["diff_ci_upper"],
            "auroc_p_bootstrap_two_sided": auc_diff["p_bootstrap_two_sided"],
            "auroc_n_bootstrap_valid": auc_diff["n_bootstrap_valid"],

            "auprc_ffq": auprc_ffq,
            "auprc_rc": auprc_rc,
            "auprc_diff_ffq_minus_rc": auprc_ffq - auprc_rc,
            "auprc_diff_boot_mean": auprc_diff["diff_mean"],
            "auprc_diff_ci_lower": auprc_diff["diff_ci_lower"],
            "auprc_diff_ci_upper": auprc_diff["diff_ci_upper"],
            "auprc_p_bootstrap_two_sided": auprc_diff["p_bootstrap_two_sided"],
            "auprc_n_bootstrap_valid": auprc_diff["n_bootstrap_valid"],
        }

        rows.append(row)

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[SUMMARY]\n")
    show_cols = [
        "target",
        "n",
        "prevalence",
        "auroc_ffq",
        "auroc_rc",
        "auroc_diff_ffq_minus_rc",
        "auroc_diff_ci_lower",
        "auroc_diff_ci_upper",
        "auroc_p_bootstrap_two_sided",
        "auprc_ffq",
        "auprc_rc",
        "auprc_diff_ffq_minus_rc",
    ]

    print(result[show_cols])

    print(f"\n[SAVED] {OUTPUT}")

if __name__ == "__main__":
    main()