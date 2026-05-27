# ============================================================
# 94_build_mutually_exclusive_predisease_groups.py
#
# Build mutually exclusive pre-disease groups:
# - isolated preDM only
# - isolated preHTN only
# - isolated borderline lipid only
# - double burden
# - triple burden
#
# Output:
# - dataset with exclusive group labels
# - N summary
# - food-group profile by group
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "58_semihealthy_burden_stratification",
    "58_dataset_with_burden.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "94_mutually_exclusive_predisease_groups"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

print("[INPUT]")
print(df.shape)

# ============================================================
# REQUIRED LABELS
# ============================================================

label_cols = {
    "preDM": "label_prediabetes_clean2",
    "preHTN": "label_prehypertension_clean2",
    "lipid": "label_borderline_lipid_clean",
}

for name, col in label_cols.items():
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# ============================================================
# BUILD EXCLUSIVE GROUPS
# ============================================================

preDM = df[label_cols["preDM"]]
preHTN = df[label_cols["preHTN"]]
lipid = df[label_cols["lipid"]]

df["predisease_burden_n"] = preDM + preHTN + lipid

df["exclusive_predisease_group"] = "unclassified"

df.loc[
    (preDM == 0) & (preHTN == 0) & (lipid == 0),
    "exclusive_predisease_group"
] = "metabolically_normal"

df.loc[
    (preDM == 1) & (preHTN == 0) & (lipid == 0),
    "exclusive_predisease_group"
] = "isolated_preDM"

df.loc[
    (preDM == 0) & (preHTN == 1) & (lipid == 0),
    "exclusive_predisease_group"
] = "isolated_preHTN"

df.loc[
    (preDM == 0) & (preHTN == 0) & (lipid == 1),
    "exclusive_predisease_group"
] = "isolated_lipid"

df.loc[
    (preDM == 1) & (preHTN == 1) & (lipid == 0),
    "exclusive_predisease_group"
] = "preDM_preHTN"

df.loc[
    (preDM == 1) & (preHTN == 0) & (lipid == 1),
    "exclusive_predisease_group"
] = "preDM_lipid"

df.loc[
    (preDM == 0) & (preHTN == 1) & (lipid == 1),
    "exclusive_predisease_group"
] = "preHTN_lipid"

df.loc[
    (preDM == 1) & (preHTN == 1) & (lipid == 1),
    "exclusive_predisease_group"
] = "triple_burden"

df["exclusive_predisease_group_simple"] = df["exclusive_predisease_group"]

df.loc[
    df["predisease_burden_n"] >= 2,
    "exclusive_predisease_group_simple"
] = "multi_burden_2plus"

# ============================================================
# GROUP ORDER
# ============================================================

group_order = [
    "metabolically_normal",
    "isolated_preDM",
    "isolated_preHTN",
    "isolated_lipid",
    "preDM_preHTN",
    "preDM_lipid",
    "preHTN_lipid",
    "triple_burden",
]

simple_order = [
    "metabolically_normal",
    "isolated_preDM",
    "isolated_preHTN",
    "isolated_lipid",
    "multi_burden_2plus",
]

# ============================================================
# SUMMARY
# ============================================================

summary = (
    df["exclusive_predisease_group"]
    .value_counts()
    .reindex(group_order)
    .reset_index()
)

summary.columns = ["exclusive_predisease_group", "n"]
summary["percent"] = summary["n"] / summary["n"].sum() * 100

simple_summary = (
    df["exclusive_predisease_group_simple"]
    .value_counts()
    .reindex(simple_order)
    .reset_index()
)

simple_summary.columns = ["exclusive_predisease_group_simple", "n"]
simple_summary["percent"] = simple_summary["n"] / simple_summary["n"].sum() * 100

print("\n[EXCLUSIVE GROUP SUMMARY]")
print(summary)

print("\n[SIMPLE GROUP SUMMARY]")
print(simple_summary)

# ============================================================
# FOOD FEATURES
# ============================================================

food_features = [
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

food_labels = {
    "fg_refined_grain_z": "Refined grain",
    "fg_whole_grain_z": "Whole grain",
    "fg_fastfood_z": "Fast food / sweets",
    "fg_meat_processed_z": "Meat / processed meat",
    "fg_fish_seafood_z": "Fish / seafood",
    "fg_vegetable_z": "Vegetables",
    "fg_kimchi_fermented_z": "Kimchi / fermented veg",
    "fg_fruit_z": "Fruit",
    "fg_dairy_z": "Dairy",
    "fg_sweet_beverage_z": "Sweet beverage",
    "fg_alcohol_z": "Alcohol",
    "fg_coffee_tea_z": "Coffee / tea",
}

available_food = [c for c in food_features if c in df.columns]

for c in available_food:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ============================================================
# FOOD PROFILE BY GROUP
# ============================================================

food_profile = (
    df.groupby("exclusive_predisease_group")[available_food]
    .mean()
    .reindex(group_order)
)

food_profile.index.name = "group"
food_profile.columns = [food_labels.get(c, c) for c in food_profile.columns]

simple_food_profile = (
    df.groupby("exclusive_predisease_group_simple")[available_food]
    .mean()
    .reindex(simple_order)
)

simple_food_profile.index.name = "group"
simple_food_profile.columns = [food_labels.get(c, c) for c in simple_food_profile.columns]

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    os.path.join(
        OUT_DIR,
        "94_dataset_with_exclusive_predisease_groups.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "94_exclusive_group_n_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

simple_summary.to_csv(
    os.path.join(
        OUT_DIR,
        "94_simple_group_n_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

food_profile.to_csv(
    os.path.join(
        OUT_DIR,
        "94_exclusive_group_food_profile.csv"
    ),
    encoding="utf-8-sig"
)

simple_food_profile.to_csv(
    os.path.join(
        OUT_DIR,
        "94_simple_group_food_profile.csv"
    ),
    encoding="utf-8-sig"
)

# ============================================================
# FIGURE 1: N BARPLOT
# ============================================================

plt.figure(figsize=(10, 4.8))

plt.bar(
    summary["exclusive_predisease_group"],
    summary["n"]
)

plt.xticks(rotation=35, ha="right")
plt.ylabel("N")
plt.title("Mutually exclusive pre-disease group size")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "94_exclusive_group_n_barplot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# FIGURE 2: SIMPLE FOOD PROFILE HEATMAP
# ============================================================

heat = simple_food_profile.copy()

# row-wise z-score across groups for each food variable
heat_z = heat.copy()

for col in heat_z.columns:
    mean = heat_z[col].mean()
    sd = heat_z[col].std()

    if sd == 0 or pd.isna(sd):
        heat_z[col] = 0
    else:
        heat_z[col] = (heat_z[col] - mean) / sd

plt.figure(figsize=(10, 5.5))

plt.imshow(
    heat_z,
    aspect="auto",
    cmap="viridis"
)

plt.colorbar(label="Across-group z-score")

plt.xticks(
    range(len(heat_z.columns)),
    heat_z.columns,
    rotation=35,
    ha="right"
)

plt.yticks(
    range(len(heat_z.index)),
    heat_z.index
)

plt.title("Food-group profile across mutually exclusive pre-disease groups")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUT_DIR,
        "94_simple_group_food_profile_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n[SAVED]")
print(OUT_DIR)