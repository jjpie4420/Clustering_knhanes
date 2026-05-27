import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
import shap

from config import PROCESSED_DIR, RESULT_DIR

warnings.filterwarnings("ignore")

INPUT = PROCESSED_DIR / "analysis_cohort_energy_adjusted.csv"
OUTPUT = RESULT_DIR / "24_shap_feature_importance.csv"

FIG_DIR = RESULT_DIR / "figures_shap"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_DISPLAY = 15

TARGETS = {
    "diabetes": "label_diabetes",
    "hypertension": "label_hypertension",
    "dyslipidemia": "label_dyslipidemia",
    "obesity": "label_obesity",
}

FFQ_ADJ = [
    "FQ_PROT_adj", "FQ_FAT_adj", "FQ_SFA_adj", "FQ_MUFA_adj",
    "FQ_PUFA_adj",
    "FQ_N3_adj", "FQ_N6_adj", "FQ_CHOL_adj", "FQ_CHO_adj", "FQ_TDF_adj",
    "FQ_CA_adj", "FQ_PHOS_adj", "FQ_FE_adj", "FQ_NA_adj", "FQ_K_adj",
    "FQ_VA_adj", "FQ_CAROT_adj", "FQ_RETIN_adj", "FQ_B1_adj", "FQ_B2_adj",
    "FQ_NIAC_adj", "FQ_VITC_adj",
]

FEATURE_LABELS = {
    "FQ_PROT_adj": "Protein",
    "FQ_FAT_adj": "Fat",
    "FQ_SFA_adj": "SFA",
    "FQ_MUFA_adj": "MUFA",
    "FQ_PUFA_adj": "PUFA",
    "FQ_N3_adj": "n-3 FA",
    "FQ_N6_adj": "n-6 FA",
    "FQ_CHOL_adj": "Cholesterol",
    "FQ_CHO_adj": "Carbohydrate",
    "FQ_TDF_adj": "Fiber",
    "FQ_CA_adj": "Calcium",
    "FQ_PHOS_adj": "Phosphorus",
    "FQ_FE_adj": "Iron",
    "FQ_NA_adj": "Sodium",
    "FQ_K_adj": "Potassium",
    "FQ_VA_adj": "Vitamin A",
    "FQ_CAROT_adj": "Carotene",
    "FQ_RETIN_adj": "Retinol",
    "FQ_B1_adj": "Thiamin",
    "FQ_B2_adj": "Riboflavin",
    "FQ_NIAC_adj": "Niacin",
    "FQ_VITC_adj": "Vitamin C",
}

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

def make_model(y_train):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model

def impute_train_test(X_train, X_test):
    imputer = SimpleImputer(strategy="median")

    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_imp, X_test_imp

def plot_shap_summary(shap_values, X_test, target_name):
    X_plot = X_test.rename(columns=FEATURE_LABELS)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_plot,
        max_display=MAX_DISPLAY,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"shap_summary_{target_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def plot_shap_bar(shap_values, X_test, target_name):
    X_plot = X_test.rename(columns=FEATURE_LABELS)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_plot,
        plot_type="bar",
        max_display=MAX_DISPLAY,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"shap_bar_{target_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def plot_dependence_top_features(shap_values, X_test, target_name, top_features):
    for feature in top_features[:5]:
        plt.figure()
        shap.dependence_plot(
            feature,
            shap_values,
            X_test,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            FIG_DIR / f"shap_dependence_{target_name}_{feature}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    features = existing(FFQ_ADJ, df)
    all_rows = []

    for target_name, label_col in TARGETS.items():
        if label_col not in df.columns:
            continue

        X, y = clean_xy(df, label_col, features)

        if len(y) < 500 or y.sum() < 50:
            print(f"[SKIP] {target_name}: insufficient sample")
            continue

        print(f"\n[TARGET] {target_name} | n={len(y)} | pos={int(y.sum())}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        X_train_imp, X_test_imp = impute_train_test(X_train, X_test)

        model = make_model(y_train)
        model.fit(X_train_imp, y_train)

        pred = model.predict_proba(X_test_imp)[:, 1]

        auc = roc_auc_score(y_test, pred)
        auprc = average_precision_score(y_test, pred)

        print(f"  XGBoost AUROC={auc:.3f}, AUPRC={auprc:.3f}")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_imp)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        mean_shap = shap_values.mean(axis=0)

        result = pd.DataFrame({
            "target": target_name,
            "feature": features,
            "feature_label": [FEATURE_LABELS.get(f, f) for f in features],
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
            "direction": np.where(mean_shap > 0, "positive", "negative"),
            "xgb_test_auroc": auc,
            "xgb_test_auprc": auprc,
            "n": len(y),
            "n_positive": int(y.sum()),
            "prevalence": y.mean(),
        }).sort_values("mean_abs_shap", ascending=False)

        all_rows.append(result)

        print(result[["feature_label", "mean_abs_shap", "mean_shap", "direction"]].head(10))

        plot_shap_summary(shap_values, X_test_imp, target_name)
        plot_shap_bar(shap_values, X_test_imp, target_name)

        top_features = result["feature"].head(5).tolist()
        plot_dependence_top_features(
            shap_values=shap_values,
            X_test=X_test_imp,
            target_name=target_name,
            top_features=top_features,
        )

    final = pd.concat(all_rows, axis=0, ignore_index=True)

    final.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    top10 = (
        final
        .sort_values(["target", "mean_abs_shap"], ascending=[True, False])
        .groupby("target")
        .head(10)
    )

    top10.to_csv(
        RESULT_DIR / "24_shap_top10_by_target.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n[SAVED] {OUTPUT}")
    print(f"[SAVED] {RESULT_DIR / '24_shap_top10_by_target.csv'}")
    print(f"[FIGURES] {FIG_DIR}")

if __name__ == "__main__":
    main()