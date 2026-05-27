import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from config import PROCESSED_DIR, RESULT_DIR

INPUT = PROCESSED_DIR / "analysis_cohort_semihealthy_foodgroups.csv"

OUT_DIR = RESULT_DIR / "31_prehypertension_foodpattern"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_SUMMARY = OUT_DIR / "31_prehypertension_foodgroup_compare.csv"
OUT_FIG = OUT_DIR / "31_prehypertension_foodgroup_difference.png"

FOOD_GROUPS = [
    "fg_refined_grain_z",
    "fg_whole_grain_z",
    "fg_fastfood_z",
    "fg_meat_processed_z",
    "fg_fish_seafood_z",
    "fg_vegetable_z",
    "fg_kimchi_fermented_z",
    "fg_fruit_z",
    "fg_dairy_z",
    "fg_sweet_beverage_z",
    "fg_alcohol_z",
    "fg_coffee_tea_z",
]

FOOD_LABELS = {
    "fg_refined_grain_z": "Refined grain",
    "fg_whole_grain_z": "Whole grain",
    "fg_fastfood_z": "Fast food / sweets",
    "fg_meat_processed_z": "Meat / processed meat",
    "fg_fish_seafood_z": "Fish / seafood",
    "fg_vegetable_z": "Vegetables",
    "fg_kimchi_fermented_z": "Kimchi / fermented veg",
    "fg_fruit_z": "Fruit",
    "fg_dairy_z": "Dairy",
    "fg_sweet_beverage_z": "Sweet beverage / sweets",
    "fg_alcohol_z": "Alcohol",
    "fg_coffee_tea_z": "Coffee / tea",
}

def cohen_d(x1, x0):
    x1 = pd.Series(x1).dropna()
    x0 = pd.Series(x0).dropna()

    n1, n0 = len(x1), len(x0)

    if n1 < 2 or n0 < 2:
        return np.nan

    s1 = x1.std()
    s0 = x0.std()

    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))

    if pooled_sd == 0:
        return np.nan

    return (x1.mean() - x0.mean()) / pooled_sd

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    print("[INPUT]")
    print(df.shape)

    # bp_group은 30번 스크립트에서 생성됨
    sub = df[df["bp_group"].isin(["normal_bp", "prehypertension_clean"])].copy()

    sub["label_prehypertension_foodpattern"] = np.where(
        sub["bp_group"] == "prehypertension_clean", 1, 0
    )

    print("\n[GROUP COUNTS]")
    print(sub["bp_group"].value_counts())

    case = sub[sub["label_prehypertension_foodpattern"] == 1]
    control = sub[sub["label_prehypertension_foodpattern"] == 0]

    print("\n[N]")
    print(f"Normal BP: {len(control)}")
    print(f"Prehypertension clean: {len(case)}")

    rows = []

    for col in FOOD_GROUPS:
        if col not in sub.columns:
            print(f"[MISSING] {col}")
            continue

        x_case = pd.to_numeric(case[col], errors="coerce")
        x_control = pd.to_numeric(control[col], errors="coerce")

        t, p = ttest_ind(
            x_case.dropna(),
            x_control.dropna(),
            equal_var=False
        )

        d = cohen_d(x_case, x_control)

        rows.append({
            "food_group": col,
            "food_label": FOOD_LABELS.get(col, col),
            "normal_mean_z": x_control.mean(),
            "prehypertension_mean_z": x_case.mean(),
            "difference_prehtn_minus_normal": x_case.mean() - x_control.mean(),
            "cohen_d": d,
            "p_value": p,
            "n_normal": x_control.notna().sum(),
            "n_prehypertension": x_case.notna().sum(),
        })

    result = pd.DataFrame(rows)

    # FDR correction
    result = result.sort_values("p_value").reset_index(drop=True)
    m = len(result)
    result["p_rank"] = np.arange(1, m + 1)
    result["fdr_bh"] = (result["p_value"] * m / result["p_rank"]).clip(upper=1)
    result = result.sort_values("difference_prehtn_minus_normal").reset_index(drop=True)

    result.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n[SUMMARY]")
    print(result[[
        "food_label",
        "normal_mean_z",
        "prehypertension_mean_z",
        "difference_prehtn_minus_normal",
        "cohen_d",
        "p_value",
        "fdr_bh"
    ]])

    # plot
    plot_df = result.copy()

    plt.figure(figsize=(8, 6))

    colors = [
        "red" if x > 0 else "blue"
        for x in plot_df["difference_prehtn_minus_normal"]
    ]

    plt.barh(
        plot_df["food_label"],
        plot_df["difference_prehtn_minus_normal"],
        color=colors
    )

    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("Prehypertension - Normal BP mean difference (Z-score)")
    plt.title("Habitual food-group pattern in prehypertension-clean adults")

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[SAVED]")
    print(OUT_SUMMARY)
    print(OUT_FIG)

if __name__ == "__main__":
    main()