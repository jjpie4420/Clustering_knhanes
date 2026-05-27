import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    balanced_accuracy_score,
)

from config import PROCESSED_DIR, RESULT_DIR

warnings.filterwarnings("ignore")

INPUT = PROCESSED_DIR / "analysis_cohort_energy_adjusted.csv"

OUTPUT = RESULT_DIR / "21_temporal_validation_results.csv"

FIG_DIR = RESULT_DIR / "figures_temporal_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

###########################################################
# TARGETS
###########################################################

TARGETS = {
    "diabetes": "label_diabetes",
    "hypertension": "label_hypertension",
    "dyslipidemia": "label_dyslipidemia",
    "obesity": "label_obesity",
}

###########################################################
# FEATURES
###########################################################

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

###########################################################
# FUNCTIONS
###########################################################

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

    tmp = df[[label_col, "year"] + features].copy()

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

    return tmp

def evaluate_test(y_true, pred_prob):

    pred_label = (pred_prob >= 0.5).astype(int)

    return {
        "auroc": roc_auc_score(y_true, pred_prob),
        "auprc": average_precision_score(y_true, pred_prob),
        "brier": brier_score_loss(y_true, pred_prob),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            pred_label
        ),
        "f1": f1_score(
            y_true,
            pred_label
        ),
    }

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

    rows = []

    #######################################################
    # temporal split
    #######################################################

    train_years = [2012, 2013, 2014]
    test_years = [2015, 2016]

    print(f"\n[TRAIN YEARS] {train_years}")
    print(f"[TEST YEARS] {test_years}")

    for target_name, label_col in TARGETS.items():

        ###################################################
        # FFQ
        ###################################################

        tmp_ffq = clean_xy(
            df,
            label_col,
            ffq_features
        )

        ###################################################
        # RC
        ###################################################

        tmp_rc = clean_xy(
            df,
            label_col,
            rc_features
        )

        ###################################################
        # common index
        ###################################################

        common_idx = tmp_ffq.index.intersection(
            tmp_rc.index
        )

        tmp_ffq = tmp_ffq.loc[common_idx]
        tmp_rc = tmp_rc.loc[common_idx]

        ###################################################
        # train/test split
        ###################################################

        train_ffq = tmp_ffq[
            tmp_ffq["year"].isin(train_years)
        ]

        test_ffq = tmp_ffq[
            tmp_ffq["year"].isin(test_years)
        ]

        train_rc = tmp_rc[
            tmp_rc["year"].isin(train_years)
        ]

        test_rc = tmp_rc[
            tmp_rc["year"].isin(test_years)
        ]

        ###################################################
        # FFQ data
        ###################################################

        X_train_ffq = train_ffq[ffq_features]
        y_train_ffq = train_ffq[label_col].astype(int)

        X_test_ffq = test_ffq[ffq_features]
        y_test_ffq = test_ffq[label_col].astype(int)

        ###################################################
        # RC data
        ###################################################

        X_train_rc = train_rc[rc_features]
        y_train_rc = train_rc[label_col].astype(int)

        X_test_rc = test_rc[rc_features]
        y_test_rc = test_rc[label_col].astype(int)

        ###################################################
        # sample check
        ###################################################

        print(
            f"\n[TARGET] {target_name}"
            f"\nFFQ train={len(y_train_ffq)}"
            f" test={len(y_test_ffq)}"
        )

        ###################################################
        # FFQ model
        ###################################################

        ffq_model = make_pipeline()

        ffq_model.fit(
            X_train_ffq,
            y_train_ffq
        )

        pred_ffq = ffq_model.predict_proba(
            X_test_ffq
        )[:, 1]

        ffq_metrics = evaluate_test(
            y_test_ffq,
            pred_ffq
        )

        ###################################################
        # RC model
        ###################################################

        rc_model = make_pipeline()

        rc_model.fit(
            X_train_rc,
            y_train_rc
        )

        pred_rc = rc_model.predict_proba(
            X_test_rc
        )[:, 1]

        rc_metrics = evaluate_test(
            y_test_rc,
            pred_rc
        )

        ###################################################
        # print
        ###################################################

        print(
            f"FFQ AUROC={ffq_metrics['auroc']:.3f}"
            f" | RC AUROC={rc_metrics['auroc']:.3f}"
            f" | Δ={ffq_metrics['auroc'] - rc_metrics['auroc']:.3f}"
        )

        ###################################################
        # save
        ###################################################

        for model_name, metrics in [
            ("ffq_adj_only", ffq_metrics),
            ("rc_adj_only", rc_metrics),
        ]:

            rows.append({
                "target": target_name,
                "model": model_name,

                "train_years": ",".join(map(str, train_years)),
                "test_years": ",".join(map(str, test_years)),

                "n_train": len(y_train_ffq),
                "n_test": len(y_test_ffq),

                "prevalence_train": y_train_ffq.mean(),
                "prevalence_test": y_test_ffq.mean(),

                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier": metrics["brier"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
            })

    #######################################################
    # result
    #######################################################

    result = pd.DataFrame(rows)

    #######################################################
    # delta table
    #######################################################

    pivot = result.pivot_table(
        index="target",
        columns="model",
        values="auroc"
    )

    pivot["delta_ffq_minus_rc"] = (
        pivot["ffq_adj_only"]
        - pivot["rc_adj_only"]
    )

    print("\n[TEMPORAL VALIDATION SUMMARY]\n")
    print(pivot)

    #######################################################
    # save
    #######################################################

    result.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    #######################################################
    # plot
    #######################################################

    plot_df = pivot.reset_index()

    x = np.arange(len(plot_df))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(
        x - width/2,
        plot_df["ffq_adj_only"],
        width,
        label="FFQ"
    )

    plt.bar(
        x + width/2,
        plot_df["rc_adj_only"],
        width,
        label="24h recall"
    )

    plt.xticks(
        x,
        plot_df["target"],
        rotation=45,
        ha="right"
    )

    plt.ylabel("Temporal validation AUROC")
    plt.ylim(0.5, 0.8)

    plt.title(
        "Temporal validation (train: 2012-2014, test: 2015-2016)"
    )

    plt.legend()

    plt.tight_layout()

    fig_path = (
        FIG_DIR
        / "temporal_validation_auc.png"
    )

    plt.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\n[SAVED] {OUTPUT}")
    print(f"[FIGURE] {fig_path}")

if __name__ == "__main__":
    main()