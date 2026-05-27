# ============================================
# 28_build_ffq_food_groups.py
# ============================================

import pandas as pd
import numpy as np
from scipy.stats import zscore

# --------------------------------------------
# LOAD
# --------------------------------------------

df = pd.read_csv(
    r"D:\precision_nutrition\FFQ\processed\analysis_cohort_energy_adjusted.csv"
)

# --------------------------------------------
# FOOD GROUP DEFINITIONS
# --------------------------------------------

food_groups = {

    # ----------------------------------------
    # Refined grains / white rice
    # ----------------------------------------
    "fg_refined_grain": [
        "FQ_RICE",
        "FQ_BIBIM",
        "FQ_GIMBAB",
        "FQ_INSTNO",
        "FQ_WHNO",
        "FQ_CHNO",
        "FQ_BUNO",
        "FQ_BRE"
    ],

    # ----------------------------------------
    # Whole grains
    # ----------------------------------------
    "fg_whole_grain": [
        "FQ_BARLEY",
        "FQ_CEREAL",
        "FQ_RB_BRE"
    ],

    # ----------------------------------------
    # Fast food / processed carb
    # ----------------------------------------
    "fg_fastfood": [
        "FQ_PIZZA",
        "FQ_HAMBER",
        "FQ_CAKE",
        "FQ_COOKIE",
        "FQ_SNACK",
        "FQ_CHOCO",
        "FQ_ICECM"
    ],

    # ----------------------------------------
    # Meat / processed meat
    # ----------------------------------------
    "fg_meat_processed": [
        "FQ_R_PORK",
        "FQ_S_PORK",
        "FQ_F_PORK",
        "FQ_C_PORK",
        "FQ_R_BEEF",
        "FQ_F_BEEF",
        "FQ_HAM",
        "FQ_PORKBY",
        "FQ_T_CHIC",
        "FQ_S_CHIC",
        "FQ_F_CHIC",
        "FQ_R_DUCK"
    ],

    # ----------------------------------------
    # Fish / seafood
    # ----------------------------------------
    "fg_fish_seafood": [
        "FQ_MACKER",
        "FQ_HAIRT",
        "FQ_ANCH",
        "FQ_SQUID",
        "FQ_SCRAB",
        "FQ_SFISH",
        "FQ_FPASTE"
    ],

    # ----------------------------------------
    # Vegetables
    # ----------------------------------------
    "fg_vegetable": [
        "FQ_SPINAC",
        "FQ_BELLFI",
        "FQ_PUMPKI",
        "FQ_OVEG",
        "FQ_CUCUMB",
        "FQ_RADI",
        "FQ_VSALAD",
        "FQ_GREENO",
        "FQ_RAWVEG",
        "FQ_BROCOL",
        "FQ_GARLIC",
        "FQ_ROOT",
        "FQ_F_VEG",
        "FQ_MUSHRO"
    ],

    # ----------------------------------------
    # Kimchi / fermented vegetables
    # ----------------------------------------
    "fg_kimchi_fermented": [
        "FQ_KIMCHI",
        "FQ_OKIMCH",
        "FQ_FER_BN",
        "FQ_SVEG"
    ],

    # ----------------------------------------
    # Fruits
    # ----------------------------------------
    "fg_fruit": [
        "FQ_STRAW",
        "FQ_TOMATO",
        "FQ_MMELON",
        "FQ_WMELON",
        "FQ_PEACH",
        "FQ_GRAPE",
        "FQ_APPLE",
        "FQ_PEAR",
        "FQ_PERS",
        "FQ_TANG",
        "FQ_BANANA",
        "FQ_CITRUS",
        "FQ_KIWI"
    ],

    # ----------------------------------------
    # Dairy
    # ----------------------------------------
    "fg_dairy": [
        "FQ_MILK",
        "FQ_L_YOGU",
        "FQ_C_YOGU",
        "FQ_BN_MILK"
    ],

    # ----------------------------------------
    # Sugary beverage / sweets
    # ----------------------------------------
    "fg_sweet_beverage": [
        "FQ_SODA",
        "FQ_FJUICE",
        "FQ_SUGAR",
        "FQ_GBAVER",
        "FQ_CHOCO",
        "FQ_COOKIE",
        "FQ_ICECM"
    ],

    # ----------------------------------------
    # Alcohol
    # ----------------------------------------
    "fg_alcohol": [
        "FQ_SOJU",
        "FQ_BEER",
        "FQ_RWINE"
    ],

    # ----------------------------------------
    # Coffee / tea
    # ----------------------------------------
    "fg_coffee_tea": [
        "FQ_COFFEE",
        "FQ_TEA",
        "FQ_CREAM",
        "FQ_SUGAR"
    ]
}

# --------------------------------------------
# CREATE FOOD GROUP SCORES
# --------------------------------------------

created_groups = []

for group_name, vars_list in food_groups.items():

    existing_vars = [v for v in vars_list if v in df.columns]

    print(f"{group_name}: {len(existing_vars)} vars")

    df[group_name] = df[existing_vars].sum(axis=1)

    created_groups.append(group_name)

# --------------------------------------------
# Z-SCORE STANDARDIZATION
# --------------------------------------------

for col in created_groups:

    df[col + "_z"] = zscore(df[col], nan_policy='omit')

# --------------------------------------------
# SAVE
# --------------------------------------------

keep_cols = ["ID"] + created_groups + [x + "_z" for x in created_groups]

out = df[keep_cols]

out.to_csv(
    r"D:\precision_nutrition\FFQ\processed\ffq_food_groups.csv",
    index=False
)

print("\nSaved: ffq_food_groups.csv")
print(out.head())