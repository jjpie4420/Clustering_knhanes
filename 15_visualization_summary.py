import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import RESULT_DIR

FIG_DIR = RESULT_DIR / "figures_summary"
FIG_DIR.mkdir(parents=True, exist_ok=True)

AUC_FILE = RESULT_DIR / "14_auc_bootstrap_ffq_vs_rc.csv"
ML_FILE = RESULT_DIR / "12_ml_energy_adjusted_results.csv"
FI_FILE = RESULT_DIR / "13_energy_adjusted_feature_importance.csv"

TARGET_ORDER = [
    "diabetes",
    "diabetes_untreated",
    "prediabetes_clean",
    "hypertension",
    "hypertension_untreated",
    "prehypertension_clean",
    "dyslipidemia",
    "dyslipidemia_untreated",
    "obesity",
]

TARGET_LABELS = {
    "diabetes": "Diabetes",
    "diabetes_untreated": "Untreated diabetes",
    "prediabetes_clean": "Prediabetes",
    "hypertension": "Hypertension",
    "hypertension_untreated": "Untreated hypertension",
    "prehypertension_clean": "Prehypertension",
    "dyslipidemia": "Dyslipidemia",
    "dyslipidemia_untreated": "Untreated dyslipidemia",
    "obesity": "Obesity",
}

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_auc_ffq_vs_rc():
    df = pd.read_csv(AUC_FILE, encoding="utf-8-sig")
    df["target_label"] = df["target"].map(TARGET_LABELS)
    df["target"] = pd.Categorical(df["target"], TARGET_ORDER, ordered=True)
    df = df.sort_values("target")

    x = range(len(df))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar([i - width/2 for i in x], df["auroc_ffq"], width, label="FFQ")
    plt.bar([i + width/2 for i in x], df["auroc_rc"], width, label="24h recall")

    plt.xticks(x, df["target_label"], rotation=45, ha="right")
    plt.ylabel("AUROC")
    plt.ylim(0.5, 0.75)
    plt.title("Energy-adjusted diet-only prediction: FFQ vs 24h recall")
    plt.legend()

    savefig(FIG_DIR / "figure_auc_ffq_vs_rc.png")

def plot_auc_difference_ci():
    df = pd.read_csv(AUC_FILE, encoding="utf-8-sig")
    df["target_label"] = df["target"].map(TARGET_LABELS)
    df["target"] = pd.Categorical(df["target"], TARGET_ORDER, ordered=True)
    df = df.sort_values("target")

    y = range(len(df))

    diff = df["auroc_diff_ffq_minus_rc"]
    lower = diff - df["auroc_diff_ci_lower"]
    upper = df["auroc_diff_ci_upper"] - diff

    plt.figure(figsize=(8, 5.5))
    plt.errorbar(
        diff,
        y,
        xerr=[lower, upper],
        fmt="o",
        capsize=4
    )
    plt.axvline(0, linestyle="--", linewidth=1)

    plt.yticks(y, df["target_label"])
    plt.xlabel("ΔAUROC: FFQ - 24h recall")
    plt.title("Bootstrap 95% CI for AUROC difference")
    plt.gca().invert_yaxis()

    savefig(FIG_DIR / "figure_auc_difference_ci.png")

def plot_auprc_ffq_vs_rc():
    df = pd.read_csv(AUC_FILE, encoding="utf-8-sig")
    df["target_label"] = df["target"].map(TARGET_LABELS)
    df["target"] = pd.Categorical(df["target"], TARGET_ORDER, ordered=True)
    df = df.sort_values("target")

    x = range(len(df))
    width = 0.35

    plt.figure(figsize=(11, 5))
    plt.bar([i - width/2 for i in x], df["auprc_ffq"], width, label="FFQ")
    plt.bar([i + width/2 for i in x], df["auprc_rc"], width, label="24h recall")

    plt.xticks(x, df["target_label"], rotation=45, ha="right")
    plt.ylabel("AUPRC")
    plt.title("Energy-adjusted diet-only prediction: AUPRC")
    plt.legend()

    savefig(FIG_DIR / "figure_auprc_ffq_vs_rc.png")

def plot_model_comparison():
    df = pd.read_csv(ML_FILE, encoding="utf-8-sig")

    df = df[
        (df["model"] == "logistic_l2") &
        (df["feature_set"].isin([
            "ffq_adj_only",
            "rc_adj_only",
            "ffq_rc_adj_combined"
        ]))
    ].copy()

    df["target_label"] = df["target"].map(TARGET_LABELS)
    df["target"] = pd.Categorical(df["target"], TARGET_ORDER, ordered=True)
    df = df.sort_values(["target", "feature_set"])

    pivot = df.pivot(
        index="target",
        columns="feature_set",
        values="roc_auc_mean"
    ).loc[TARGET_ORDER]

    labels = [TARGET_LABELS[t] for t in pivot.index]

    x = range(len(pivot))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar([i - width for i in x], pivot["ffq_adj_only"], width, label="FFQ")
    plt.bar([i for i in x], pivot["rc_adj_only"], width, label="24h recall")
    plt.bar([i + width for i in x], pivot["ffq_rc_adj_combined"], width, label="Combined")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("AUROC")
    plt.ylim(0.5, 0.75)
    plt.title("Energy-adjusted diet-only logistic models")
    plt.legend()

    savefig(FIG_DIR / "figure_model_comparison_ffq_rc_combined.png")

def plot_feature_importance_heatmap():
    df = pd.read_csv(FI_FILE, encoding="utf-8-sig")

    df = df[
        (df["feature_set"] == "ffq_adj_only") &
        (df["target"].isin([
            "diabetes",
            "prediabetes_clean",
            "hypertension",
            "dyslipidemia",
            "obesity"
        ]))
    ].copy()

    top_features = (
        df.groupby("feature")["abs_coef_mean"]
        .mean()
        .sort_values(ascending=False)
        .head(12)
        .index
        .tolist()
    )

    plot_df = df[df["feature"].isin(top_features)]

    pivot = plot_df.pivot(
        index="feature",
        columns="target",
        values="coef_mean"
    )

    pivot = pivot[[
        "diabetes",
        "prediabetes_clean",
        "hypertension",
        "dyslipidemia",
        "obesity",
    ]]

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot, aspect="auto")
    plt.colorbar(label="Standardized logistic coefficient")

    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(
        range(len(pivot.columns)),
        [TARGET_LABELS[c] for c in pivot.columns],
        rotation=45,
        ha="right"
    )

    plt.title("Top FFQ energy-adjusted nutrient coefficients")

    savefig(FIG_DIR / "figure_ffq_feature_importance_heatmap.png")

def export_summary_tables():
    auc = pd.read_csv(AUC_FILE, encoding="utf-8-sig")

    auc["target_label"] = auc["target"].map(TARGET_LABELS)

    keep = [
        "target",
        "target_label",
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

    auc[keep].to_csv(
        RESULT_DIR / "15_table_auc_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    fi = pd.read_csv(FI_FILE, encoding="utf-8-sig")

    top_fi = (
        fi[fi["feature_set"] == "ffq_adj_only"]
        .sort_values(["target", "abs_coef_mean"], ascending=[True, False])
        .groupby("target")
        .head(10)
    )

    top_fi.to_csv(
        RESULT_DIR / "15_table_top10_ffq_features.csv",
        index=False,
        encoding="utf-8-sig"
    )

def main():
    plot_auc_ffq_vs_rc()
    plot_auc_difference_ci()
    plot_auprc_ffq_vs_rc()
    plot_model_comparison()
    plot_feature_importance_heatmap()
    export_summary_tables()

    print(f"[SAVED FIGURES] {FIG_DIR}")
    print("[SAVED TABLES]")
    print(RESULT_DIR / "15_table_auc_summary.csv")
    print(RESULT_DIR / "15_table_top10_ffq_features.csv")

if __name__ == "__main__":
    main()