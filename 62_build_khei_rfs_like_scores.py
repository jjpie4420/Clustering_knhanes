import os
import numpy as np
import pandas as pd

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR,
    "results",
    "55_guideline_diet_scores",
    "55_guideline_diet_scores_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "62_khei_rfs_like_scores"
)

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

# --------------------------------------------------
# Column mapping
# --------------------------------------------------

features = {
    "fruit": "fg_fruit_z",
    "vegetable": "fg_vegetable_z",
    "whole_grain": "fg_whole_grain_z",
    "refined_grain": "fg_refined_grain_z",
    "fish": "fg_fish_seafood_z",
    "dairy": "fg_dairy_z",
    "meat_processed": "fg_meat_processed_z",
    "sweet_beverage": "fg_sweet_beverage_z",
    "fastfood": "fg_fastfood_z",
    "kimchi": "fg_kimchi_fermented_z",
    "alcohol": "fg_alcohol_z",
    "coffee_tea": "fg_coffee_tea_z",
    "sodium": "N_NA",
    "energy": "FQ_EN",
}

for k, c in features.items():
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        print(f"[WARNING] missing: {k} -> {c}")

def qscore_positive(x, max_score=10):
    """
    Higher intake = better score.
    Quintile based: Q1~Q5 mapped to 0~max_score.
    """
    x = pd.to_numeric(x, errors="coerce")
    q = pd.qcut(
        x.rank(method="first"),
        q=5,
        labels=[0, 0.25, 0.5, 0.75, 1],
        duplicates="drop"
    )
    return pd.to_numeric(q, errors="coerce") * max_score

def qscore_negative(x, max_score=10):
    """
    Higher intake = worse score.
    Q1 gets max, Q5 gets 0.
    """
    x = pd.to_numeric(x, errors="coerce")
    q = pd.qcut(
        x.rank(method="first"),
        q=5,
        labels=[1, 0.75, 0.5, 0.25, 0],
        duplicates="drop"
    )
    return pd.to_numeric(q, errors="coerce") * max_score

def moderate_score(x, max_score=5):

    x = pd.to_numeric(x, errors="coerce")

    q = pd.qcut(
        x.rank(method="first"),
        q=5,
        labels=False,
        duplicates="drop"
    )

    mapping = {
        0: 0.5,
        1: 1.0,
        2: 1.0,
        3: 0.5,
        4: 0.0
    }

    score = pd.Series(q).map(mapping)

    return score * max_score

def frequency_binary_from_z(x, threshold=0):
    """
    FFQ z-score 기반 rough binary.
    z >= 0이면 population average 이상 섭취로 간주.
    """
    x = pd.to_numeric(x, errors="coerce")
    return np.where(x >= threshold, 1, 0)

# --------------------------------------------------
# 1. KHEI-like score
# --------------------------------------------------
# KHEI 원래 구조:
# Adequacy + Moderation + Balance.
# 현재 데이터에서는 FFQ food-group + sodium 기반 adapted KHEI-like score로 구성.

khei = pd.DataFrame(index=df.index)

# Adequacy domain
khei["fruit"] = qscore_positive(df[features["fruit"]], max_score=5)
khei["vegetable"] = qscore_positive(df[features["vegetable"]], max_score=5)
khei["whole_grain"] = qscore_positive(df[features["whole_grain"]], max_score=5)
khei["fish_seafood"] = qscore_positive(df[features["fish"]], max_score=5)
khei["dairy"] = qscore_positive(df[features["dairy"]], max_score=10)

# Korean-specific: fermented vegetables
# 김치는 건강 식품군이지만 sodium context가 강하므로 moderate score로 처리
khei["kimchi_moderate"] = moderate_score(df[features["kimchi"]], max_score=5)

# Moderation domain
khei["refined_grain_reverse"] = qscore_negative(df[features["refined_grain"]], max_score=10)
khei["sweet_beverage_reverse"] = qscore_negative(df[features["sweet_beverage"]], max_score=10)
khei["fastfood_reverse"] = qscore_negative(df[features["fastfood"]], max_score=10)
khei["processed_meat_reverse"] = qscore_negative(df[features["meat_processed"]], max_score=10)

if features["sodium"] in df.columns:
    khei["sodium_reverse"] = qscore_negative(df[features["sodium"]], max_score=10)

# Optional alcohol moderation
khei["alcohol_moderate"] = moderate_score(df[features["alcohol"]], max_score=5)

# Normalize to 100
raw_max = khei.max(axis=0).sum()
df["score_KHEI_like"] = khei.sum(axis=1, skipna=True) / raw_max * 100

# --------------------------------------------------
# 2. Korean RFS-like score
# --------------------------------------------------
# RFS는 권장식품을 일정 빈도 이상 섭취하면 1점.
# 현재 FFQ z-score에서 평균 이상 섭취를 "regular intake"로 근사.

rfs = pd.DataFrame(index=df.index)

rfs["fruit"] = frequency_binary_from_z(df[features["fruit"]])
rfs["vegetable"] = frequency_binary_from_z(df[features["vegetable"]])
rfs["whole_grain"] = frequency_binary_from_z(df[features["whole_grain"]])
rfs["fish_seafood"] = frequency_binary_from_z(df[features["fish"]])
rfs["dairy"] = frequency_binary_from_z(df[features["dairy"]])

# RFS에서는 kimchi를 넣을지 논란 있음.
# 한국형 식이 맥락을 보존하기 위해 별도 버전으로 계산.
rfs["kimchi_fermented"] = frequency_binary_from_z(df[features["kimchi"]])

df["score_RFS5_like"] = rfs[[
    "fruit",
    "vegetable",
    "whole_grain",
    "fish_seafood",
    "dairy"
]].sum(axis=1)

df["score_RFS6_korean_like"] = rfs.sum(axis=1)

# --------------------------------------------------
# 3. Quantile-based Korean healthy food score
# --------------------------------------------------
# Median binary보다 정보 손실이 적은 버전.
# Healthy components는 quintile positive, unhealthy는 reverse.

khfs = pd.DataFrame(index=df.index)

khfs["fruit"] = qscore_positive(df[features["fruit"]], max_score=10)
khfs["vegetable"] = qscore_positive(df[features["vegetable"]], max_score=10)
khfs["whole_grain"] = qscore_positive(df[features["whole_grain"]], max_score=10)
khfs["fish"] = qscore_positive(df[features["fish"]], max_score=10)
khfs["dairy"] = qscore_positive(df[features["dairy"]], max_score=10)
khfs["refined_grain_reverse"] = qscore_negative(df[features["refined_grain"]], max_score=10)
khfs["sweet_beverage_reverse"] = qscore_negative(df[features["sweet_beverage"]], max_score=10)
khfs["fastfood_reverse"] = qscore_negative(df[features["fastfood"]], max_score=10)
khfs["processed_meat_reverse"] = qscore_negative(df[features["meat_processed"]], max_score=10)

df["score_Korean_HFS_quantile"] = khfs.mean(axis=1, skipna=True) * 10

# --------------------------------------------------
# Save components and dataset
# --------------------------------------------------

score_cols = [
    "score_KHEI_like",
    "score_RFS5_like",
    "score_RFS6_korean_like",
    "score_Korean_HFS_quantile",
]

df.to_csv(
    os.path.join(OUT_DIR, "62_dataset_with_khei_rfs_like_scores.csv"),
    index=False,
    encoding="utf-8-sig"
)

khei.to_csv(
    os.path.join(OUT_DIR, "62_khei_like_components.csv"),
    index=False,
    encoding="utf-8-sig"
)

rfs.to_csv(
    os.path.join(OUT_DIR, "62_rfs_like_components.csv"),
    index=False,
    encoding="utf-8-sig"
)

khfs.to_csv(
    os.path.join(OUT_DIR, "62_korean_hfs_quantile_components.csv"),
    index=False,
    encoding="utf-8-sig"
)

summary = df[score_cols].describe().T

summary.to_csv(
    os.path.join(OUT_DIR, "62_khei_rfs_like_score_summary.csv"),
    encoding="utf-8-sig"
)

print("[SCORE SUMMARY]")
print(summary)

print("\n[SAVED]")
print(OUT_DIR)