import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from config import RESULT_DIR

###########################################################
# PATHS
###########################################################

FIG_DIR = RESULT_DIR / "publication_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

AGREE_FILE = RESULT_DIR / "07_ffq_rc_agreement_summary.csv"
AUC_FILE = RESULT_DIR / "14_auc_bootstrap_ffq_vs_rc.csv"
CAL_FILE = RESULT_DIR / "20_calibration_results.csv"
TEMP_FILE = RESULT_DIR / "21_temporal_validation_results.csv"
SUBGROUP_FILE = RESULT_DIR / "17_subgroup_auc_ffq_vs_rc.csv"

###########################################################
# GLOBAL STYLE
###########################################################

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.2

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

###########################################################
# SAVE FUNCTION
###########################################################

def savefig(name):
    path_png = FIG_DIR / f"{name}.png"
    path_pdf = FIG_DIR / f"{name}.pdf"

    plt.tight_layout()

    plt.savefig(
        path_png,
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        path_pdf,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVED] {path_png}")

###########################################################
# FIGURE 1
###########################################################

def figure1_agreement():

    df = pd.read_csv(
        AGREE_FILE,
        encoding="utf-8-sig"
    )

    df = df.sort_values(
        "pearson_r",
        ascending=False
    )

    plt.figure(figsize=(7, 7))

    y = np.arange(len(df))

    plt.barh(
        y,
        df["pearson_r"]
    )

    plt.yticks(
        y,
        df["nutrient"]
    )

    plt.xlabel("Pearson correlation")
    plt.title(
        "Figure 1. Agreement between FFQ and 24-hour recall nutrients"
    )

    plt.gca().invert_yaxis()

    savefig("Figure1_agreement")

###########################################################
# FIGURE 2
###########################################################

def figure2_auc():

    df = pd.read_csv(
        AUC_FILE,
        encoding="utf-8-sig"
    )

    keep = [
        "diabetes",
        "hypertension",
        "dyslipidemia",
        "obesity",
    ]

    df = df[df["target"].isin(keep)].copy()

    df["label"] = df["target"].map(
        TARGET_LABELS
    )

    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(7, 5))

    plt.bar(
        x - width/2,
        df["auroc_ffq"],
        width,
        label="FFQ"
    )

    plt.bar(
        x + width/2,
        df["auroc_rc"],
        width,
        label="24h recall"
    )

    plt.xticks(
        x,
        df["label"]
    )

    plt.ylabel("AUROC")
    plt.ylim(0.5, 0.8)

    plt.title(
        "Figure 2. Predictive performance of FFQ and recall"
    )

    plt.legend(frameon=False)

    savefig("Figure2_AUROC")

###########################################################
# FIGURE 3
###########################################################

def figure3_bootstrap_ci():

    df = pd.read_csv(
        AUC_FILE,
        encoding="utf-8-sig"
    )

    keep = [
        "diabetes",
        "hypertension",
        "dyslipidemia",
        "obesity",
    ]

    df = df[df["target"].isin(keep)].copy()

    df["label"] = df["target"].map(
        TARGET_LABELS
    )

    y = np.arange(len(df))

    diff = df["auroc_diff_ffq_minus_rc"]

    lower = (
        diff
        - df["auroc_diff_ci_lower"]
    )

    upper = (
        df["auroc_diff_ci_upper"]
        - diff
    )

    plt.figure(figsize=(6, 4.5))

    plt.errorbar(
        diff,
        y,
        xerr=[lower, upper],
        fmt="o",
        capsize=4
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.yticks(
        y,
        df["label"]
    )

    plt.xlabel("ΔAUROC (FFQ - recall)")

    plt.title(
        "Figure 3. Bootstrap confidence intervals"
    )

    plt.gca().invert_yaxis()

    savefig("Figure3_bootstrap_CI")

###########################################################
# FIGURE 4
###########################################################

def figure4_calibration():

    df = pd.read_csv(
        CAL_FILE,
        encoding="utf-8-sig"
    )

    keep = [
        "diabetes",
        "hypertension",
        "dyslipidemia",
        "obesity",
    ]

    df = df[df["target"].isin(keep)].copy()

    pivot = df.pivot_table(
        index="target",
        columns="model",
        values="ece"
    )

    pivot["delta"] = (
        pivot["rc_adj_only"]
        - pivot["ffq_adj_only"]
    )

    pivot = pivot.reset_index()

    plt.figure(figsize=(7, 4.5))

    plt.bar(
        pivot["target"],
        pivot["delta"]
    )

    plt.ylabel(
        "ECE improvement (RC - FFQ)"
    )

    plt.title(
        "Figure 4. Calibration improvement of FFQ models"
    )

    plt.xticks(rotation=20)

    savefig("Figure4_calibration")

###########################################################
# FIGURE 5
###########################################################

def figure5_temporal():

    df = pd.read_csv(
        TEMP_FILE,
        encoding="utf-8-sig"
    )

    pivot = df.pivot_table(
        index="target",
        columns="model",
        values="auroc"
    )

    pivot = pivot.reset_index()

    x = np.arange(len(pivot))
    width = 0.35

    plt.figure(figsize=(7, 5))

    plt.bar(
        x - width/2,
        pivot["ffq_adj_only"],
        width,
        label="FFQ"
    )

    plt.bar(
        x + width/2,
        pivot["rc_adj_only"],
        width,
        label="24h recall"
    )

    plt.xticks(
        x,
        pivot["target"],
        rotation=20
    )

    plt.ylabel("Temporal validation AUROC")

    plt.title(
        "Figure 5. Temporal validation"
    )

    plt.legend(frameon=False)

    savefig("Figure5_temporal_validation")

###########################################################
# FIGURE 6
###########################################################

def figure6_subgroup():

    df = pd.read_csv(
        SUBGROUP_FILE,
        encoding="utf-8-sig"
    )

    df = df[
        (df["target"] == "diabetes")
    ].copy()

    df["subgroup"] = (
        df["subgroup_var"]
        + ": "
        + df["subgroup_value"].astype(str)
    )

    plt.figure(figsize=(8, 5))

    plt.barh(
        df["subgroup"],
        df["auroc_diff_ffq_minus_rc"]
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("ΔAUROC (FFQ - recall)")

    plt.title(
        "Figure 6. Subgroup robustness (diabetes)"
    )

    plt.gca().invert_yaxis()

    savefig("Figure6_subgroup")

###########################################################
# MAIN
###########################################################

def main():

    figure1_agreement()
    figure2_auc()
    figure3_bootstrap_ci()
    figure4_calibration()
    figure5_temporal()
    figure6_subgroup()

    print("\n[ALL PUBLICATION FIGURES SAVED]")
    print(FIG_DIR)

if __name__ == "__main__":
    main()