import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

from config import PROCESSED_DIR, RESULT_DIR

INPUT = PROCESSED_DIR / "analysis_cohort_2012_2016.csv"
FIG_DIR = RESULT_DIR / "figures_agreement"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAIR_NUTRIENTS = {
    "Energy": ("FQ_EN", "RC_EN"),
    "Protein": ("FQ_PROT", "RC_PROT"),
    "Fat": ("FQ_FAT", "RC_FAT"),
    "SFA": ("FQ_SFA", "RC_SFA"),
    "MUFA": ("FQ_MUFA", "RC_MUFA"),
    "PUFA": ("FQ_PUFA", "RC_PUFA"),
    "n-3": ("FQ_N3", "RC_N3"),
    "n-6": ("FQ_N6", "RC_N6"),
    "Cholesterol": ("FQ_CHOL", "RC_CHOL"),
    "Carbohydrate": ("FQ_CHO", "RC_CHO"),
    "Fiber": ("FQ_TDF", "RC_TDF"),
    "Calcium": ("FQ_CA", "RC_CA"),
    "Phosphorus": ("FQ_PHOS", "RC_PHOS"),
    "Iron": ("FQ_FE", "RC_FE"),
    "Sodium": ("FQ_NA", "RC_NA"),
    "Potassium": ("FQ_K", "RC_K"),
    "Vitamin A": ("FQ_VA", "RC_VA"),
    "Carotene": ("FQ_CAROT", "RC_CAROT"),
    "Retinol": ("FQ_RETIN", "RC_RETIN"),
    "Thiamin": ("FQ_B1", "RC_B1"),
    "Riboflavin": ("FQ_B2", "RC_B2"),
    "Niacin": ("FQ_NIAC", "RC_NIAC"),
    "Vitamin C": ("FQ_VITC", "RC_VITC"),
}

def safe_corr(x, y):
    tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 30:
        return np.nan, np.nan, np.nan, np.nan, len(tmp)

    pearson_r, pearson_p = pearsonr(tmp["x"], tmp["y"])
    spearman_r, spearman_p = spearmanr(tmp["x"], tmp["y"])
    return pearson_r, pearson_p, spearman_r, spearman_p, len(tmp)

def scatter_plot(df, nutrient, fq_col, rc_col):
    tmp = df[[fq_col, rc_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) == 0:
        return

    plt.figure(figsize=(6, 5))
    plt.scatter(tmp[fq_col], tmp[rc_col], alpha=0.25, s=8)
    plt.xlabel(f"FFQ: {fq_col}")
    plt.ylabel(f"Recall: {rc_col}")
    plt.title(f"FFQ vs 24h Recall: {nutrient}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"scatter_{nutrient.replace(' ', '_').replace('/', '_')}.png", dpi=300)
    plt.close()

def bland_altman_plot(df, nutrient, fq_col, rc_col):
    tmp = df[[fq_col, rc_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) == 0:
        return

    mean_val = tmp[[fq_col, rc_col]].mean(axis=1)
    diff_val = tmp[fq_col] - tmp[rc_col]

    mean_diff = diff_val.mean()
    sd_diff = diff_val.std()

    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff

    plt.figure(figsize=(6, 5))
    plt.scatter(mean_val, diff_val, alpha=0.25, s=8)
    plt.axhline(mean_diff, linestyle="--", linewidth=1)
    plt.axhline(upper, linestyle="--", linewidth=1)
    plt.axhline(lower, linestyle="--", linewidth=1)
    plt.xlabel("Mean of FFQ and Recall")
    plt.ylabel("FFQ - Recall")
    plt.title(f"Bland-Altman: {nutrient}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"bland_altman_{nutrient.replace(' ', '_').replace('/', '_')}.png", dpi=300)
    plt.close()

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    rows = []

    for nutrient, (fq_col, rc_col) in PAIR_NUTRIENTS.items():
        if fq_col not in df.columns or rc_col not in df.columns:
            rows.append({
                "nutrient": nutrient,
                "ffq_col": fq_col,
                "rc_col": rc_col,
                "available": False,
                "n": 0,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
                "ffq_mean": np.nan,
                "rc_mean": np.nan,
                "ffq_median": np.nan,
                "rc_median": np.nan,
                "mean_difference_ffq_minus_rc": np.nan,
            })
            continue

        df[fq_col] = pd.to_numeric(df[fq_col], errors="coerce")
        df[rc_col] = pd.to_numeric(df[rc_col], errors="coerce")

        tmp = df[[fq_col, rc_col]].replace([np.inf, -np.inf], np.nan).dropna()

        pearson_r, pearson_p, spearman_r, spearman_p, n = safe_corr(tmp[fq_col], tmp[rc_col])

        rows.append({
            "nutrient": nutrient,
            "ffq_col": fq_col,
            "rc_col": rc_col,
            "available": True,
            "n": n,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "ffq_mean": tmp[fq_col].mean(),
            "rc_mean": tmp[rc_col].mean(),
            "ffq_median": tmp[fq_col].median(),
            "rc_median": tmp[rc_col].median(),
            "mean_difference_ffq_minus_rc": (tmp[fq_col] - tmp[rc_col]).mean(),
        })

        scatter_plot(df, nutrient, fq_col, rc_col)
        bland_altman_plot(df, nutrient, fq_col, rc_col)

    result = pd.DataFrame(rows)
    result = result.sort_values("spearman_r", ascending=False)

    out_csv = RESULT_DIR / "07_ffq_rc_agreement_summary.csv"
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n[AGREEMENT SUMMARY]")
    print(result[[
        "nutrient", "n", "pearson_r", "spearman_r",
        "ffq_mean", "rc_mean", "mean_difference_ffq_minus_rc"
    ]])

    print(f"\n[SAVED] {out_csv}")
    print(f"[FIGURES] {FIG_DIR}")

if __name__ == "__main__":
    main()