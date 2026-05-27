import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
from sklearn.calibration import calibration_curve

from config import PROCESSED_DIR, RESULT_DIR

warnings.filterwarnings("ignore")

INPUT = PROCESSED_DIR / "analysis_cohort_energy_adjusted.csv"
OUTPUT = RESULT_DIR / "20_calibration_results.csv"
FIG_DIR = RESULT_DIR / "figures_calibration"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_BINS = 10

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

def get_oof_predictions(X, y):
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())

    if len(y) < 300 or n_pos < 30 or n_neg < 30:
        return None

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

    return pred

def calibration_slope_intercept(y, pred):
    eps = 1e-6
    pred = np.clip(pred, eps, 1 - eps)
    logit_pred = np.log(pred / (1 - pred))

    # calibration model: y ~ logit(pred)
    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
    )
    model.fit(logit_pred.reshape(-1, 1), y)

    intercept = model.intercept_[0]
    slope = model.coef_[0][0]

    return intercept, slope

def expected_calibration_error(y, pred, n_bins=10):
    df = pd.DataFrame({"y": y, "pred": pred})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")

    ece = 0.0
    rows = []

    for bin_name, g in df.groupby("bin", observed=True):
        n = len(g)
        mean_pred = g["pred"].mean()
        obs_rate = g["y"].mean()
        weight = n / len(df)

        ece += weight * abs(obs_rate - mean_pred)

        rows.append({
            "bin": str(bin_name),
            "n": n,
            "mean_pred": mean_pred,
            "obs_rate": obs_rate,
            "abs_error": abs(obs_rate - mean_pred),
        })

    return ece, pd.DataFrame(rows)

def plot_calibration(y, pred_ffq, pred_rc, target_name):
    plt.figure(figsize=(6, 6))

    for pred, label in [
        (pred_ffq, "FFQ"),
        (pred_rc, "24h recall"),
    ]:
        prob_true, prob_pred = calibration_curve(
            y,
            pred,
            n_bins=N_BINS,
            strategy="quantile",
        )
        plt.plot(prob_pred, prob_true, marker="o", label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title(f"Calibration curve: {target_name}")
    plt.legend()
    plt.tight_layout()

    out = FIG_DIR / f"calibration_{target_name}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    ffq_features = existing(FFQ_ADJ, df)
    rc_features = existing(RC_ADJ, df)

    rows = []
    bin_rows = []

    for target_name, label_col in TARGETS.items():
        if label_col not in df.columns:
            continue

        X_ffq, y = clean_xy(df, label_col, ffq_features)
        X_rc, y_rc = clean_xy(df, label_col, rc_features)

        common_idx = y.index.intersection(y_rc.index)
        X_ffq = X_ffq.loc[common_idx]
        X_rc = X_rc.loc[common_idx]
        y = y.loc[common_idx]

        print(f"\n[TARGET] {target_name} | n={len(y)} | pos={int(y.sum())}")

        pred_ffq = get_oof_predictions(X_ffq, y)
        pred_rc = get_oof_predictions(X_rc, y)

        if pred_ffq is None or pred_rc is None:
            print("  - skipped: insufficient sample")
            continue

        plot_calibration(y, pred_ffq, pred_rc, target_name)

        for model_name, pred in [
            ("ffq_adj_only", pred_ffq),
            ("rc_adj_only", pred_rc),
        ]:
            brier = brier_score_loss(y, pred)
            ll = log_loss(y, pred)
            auc = roc_auc_score(y, pred)
            auprc = average_precision_score(y, pred)
            intercept, slope = calibration_slope_intercept(y, pred)
            ece, bins = expected_calibration_error(y, pred, n_bins=N_BINS)

            rows.append({
                "target": target_name,
                "label_col": label_col,
                "model": model_name,
                "n": len(y),
                "n_positive": int(y.sum()),
                "prevalence": y.mean(),
                "auroc": auc,
                "auprc": auprc,
                "brier_score": brier,
                "log_loss": ll,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "ece": ece,
            })

            bins.insert(0, "target", target_name)
            bins.insert(1, "model", model_name)
            bin_rows.append(bins)

            print(
                f"  - {model_name}: "
                f"AUROC={auc:.3f}, Brier={brier:.3f}, "
                f"Slope={slope:.3f}, ECE={ece:.3f}"
            )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    if len(bin_rows) > 0:
        bin_df = pd.concat(bin_rows, ignore_index=True)
        bin_df.to_csv(
            RESULT_DIR / "20_calibration_bins.csv",
            index=False,
            encoding="utf-8-sig"
        )

    print("\n[SUMMARY]")
    print(result[[
        "target", "model", "auroc", "auprc",
        "brier_score", "log_loss",
        "calibration_intercept", "calibration_slope", "ece"
    ]])

    print(f"\n[SAVED] {OUTPUT}")
    print(f"[SAVED] {RESULT_DIR / '20_calibration_bins.csv'}")
    print(f"[FIGURES] {FIG_DIR}")

if __name__ == "__main__":
    main()