import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

INPUT = r"D:\precision_nutrition\FFQ\processed\analysis_cohort_semihealthy_foodgroups.csv"

OUT_DIR = r"D:\precision_nutrition\FFQ\results\43_prediabetes_foodpattern"

os.makedirs(OUT_DIR, exist_ok=True)

OUT_SUMMARY = f"{OUT_DIR}/43_prediabetes_foodgroup_compare.csv"
OUT_FIG = f"{OUT_DIR}/43_prediabetes_foodgroup_difference.png"

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

    pooled_sd = np.sqrt(
        ((n1 - 1) * s1**2 + (n0 - 1) * s0**2)
        / (n1 + n0 - 2)
    )

    if pooled_sd == 0:
        return np.nan

    return (x1.mean() - x0.mean()) / pooled_sd

def bh_fdr(p_values):

    p = np.asarray(p_values, dtype=float)
    n = len(p)

    order = np.argsort(p)
    ranked = np.empty(n, dtype=float)

    prev = 1.0

    for i in range(n - 1, -1, -1):

        rank = i + 1

        val = p[order[i]] * n / rank

        prev = min(prev, val)

        ranked[order[i]] = prev

    return np.clip(ranked, 0, 1)

# --------------------------------------------------
# LOAD
# --------------------------------------------------

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

if "glucose_group" not in df.columns:
    raise ValueError(
        "glucose_group column not found. "
        "Run semihealthy group-definition script first."
    )

# --------------------------------------------------
# GROUP
# --------------------------------------------------

sub = df[df["glucose_group"].isin([
    "normal_glucose",
    "prediabetes_clean"
])].copy()

sub["label_predm_foodpattern"] = np.where(
    sub["glucose_group"] == "prediabetes_clean",
    1,
    0
)

print("\n[GROUP COUNTS]")
print(sub["glucose_group"].value_counts())

case = sub[sub["label_predm_foodpattern"] == 1]
control = sub[sub["label_predm_foodpattern"] == 0]

print("\n[N]")
print(f"Normal glucose: {len(control)}")
print(f"Prediabetes clean: {len(case)}")

# --------------------------------------------------
# TEST
# --------------------------------------------------

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
        "predm_mean_z": x_case.mean(),
        "difference_predm_minus_normal":
            x_case.mean() - x_control.mean(),
        "cohen_d": d,
        "p_value": p,
        "n_normal": x_control.notna().sum(),
        "n_predm": x_case.notna().sum(),
    })

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

result = pd.DataFrame(rows)

result["fdr_bh"] = bh_fdr(result["p_value"])

result = result.sort_values(
    "difference_predm_minus_normal"
).reset_index(drop=True)

result.to_csv(
    OUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)

print("\n[SUMMARY]")
print(result[[
    "food_label",
    "normal_mean_z",
    "predm_mean_z",
    "difference_predm_minus_normal",
    "cohen_d",
    "p_value",
    "fdr_bh"
]])

# --------------------------------------------------
# FIGURE
# --------------------------------------------------

plot_df = result.copy()

plt.figure(figsize=(8, 6))

colors = [
    "red" if x > 0 else "blue"
    for x in plot_df["difference_predm_minus_normal"]
]

plt.barh(
    plot_df["food_label"],
    plot_df["difference_predm_minus_normal"],
    color=colors
)

plt.axvline(0, linestyle="--", linewidth=1)

plt.xlabel(
    "Prediabetes - Normal glucose mean difference (Z-score)"
)

plt.title(
    "Habitual food-group pattern in prediabetes-clean adults"
)

plt.tight_layout()

plt.savefig(
    OUT_FIG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n[SAVED]")
print(OUT_SUMMARY)
print(OUT_FIG)