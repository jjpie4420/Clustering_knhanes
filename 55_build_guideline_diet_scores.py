import os
import numpy as np
import pandas as pd

BASE_DIR = r"D:\precision_nutrition\FFQ"

INPUT = os.path.join(
    BASE_DIR, "processed", "analysis_cohort_semihealthy_foodgroups.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR, "results", "55_guideline_diet_scores"
)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

# =====================================================
# 변수 후보 매핑
# 네 데이터 컬럼명에 맞게 여기만 수정하면 됨
# =====================================================

VAR = {
    "energy": ["FQ_EN", "energy", "EN", "kcal"],
    "sex": ["sex"],
    "age": ["age"],

    # food-group z-score 또는 raw intake
    "fruit": ["fg_fruit_z", "fruit", "F_FRUIT"],
    "vegetable": ["fg_vegetable_z", "vegetable", "F_VEG"],
    "whole_grain": ["fg_whole_grain_z", "whole_grain"],
    "refined_grain": ["fg_refined_grain_z", "refined_grain"],
    "fish": ["fg_fish_seafood_z", "fish", "seafood"],
    "dairy": ["fg_dairy_z", "dairy"],
    "meat_processed": ["fg_meat_processed_z", "processed_meat", "red_meat"],
    "sweet_beverage": ["fg_sweet_beverage_z", "sweet_beverage", "SSB"],
    "fastfood": ["fg_fastfood_z", "fastfood"],
    "alcohol": ["fg_alcohol_z", "alcohol", "DRINK"],
    "coffee_tea": ["fg_coffee_tea_z", "coffee_tea"],
    "kimchi": ["fg_kimchi_fermented_z", "kimchi"],

    # nutrient variables if available
    "sodium": ["Sodium", "sodium", "NA", "N_NA"],
    "sfa": ["SFA", "saturated_fat", "N_SFA"],
    "pufa": ["PUFA", "N_PUFA"],
    "mufa": ["MUFA", "N_MUFA"],
    "fiber": ["Fiber", "fiber", "N_FIBER"],
}

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

COL = {k: pick_col(df, v) for k, v in VAR.items()}

print("[COLUMN MAPPING]")
for k, v in COL.items():
    print(f"{k}: {v}")

def get_series(df, key):
    col = COL.get(key)
    if col is None:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")

def median_score_positive(x):
    med = x.median(skipna=True)
    return np.where(x >= med, 1, 0)

def median_score_negative(x):
    med = x.median(skipna=True)
    return np.where(x < med, 1, 0)

def quintile_score_positive(x):
    q = pd.qcut(x.rank(method="first"), 5, labels=[1,2,3,4,5])
    return pd.to_numeric(q, errors="coerce")

def quintile_score_negative(x):
    q = pd.qcut(x.rank(method="first"), 5, labels=[5,4,3,2,1])
    return pd.to_numeric(q, errors="coerce")

def ten_point_positive(x):
    pct = x.rank(pct=True)
    return pct * 10

def ten_point_negative(x):
    pct = x.rank(pct=True)
    return (1 - pct) * 10

# =====================================================
# Core variables
# =====================================================

fruit = get_series(df, "fruit")
vegetable = get_series(df, "vegetable")
whole_grain = get_series(df, "whole_grain")
refined_grain = get_series(df, "refined_grain")
fish = get_series(df, "fish")
dairy = get_series(df, "dairy")
meat_processed = get_series(df, "meat_processed")
sweet_bev = get_series(df, "sweet_beverage")
fastfood = get_series(df, "fastfood")
alcohol = get_series(df, "alcohol")
sodium = get_series(df, "sodium")
sfa = get_series(df, "sfa")
pufa = get_series(df, "pufa")
mufa = get_series(df, "mufa")
fiber = get_series(df, "fiber")

# =====================================================
# 1) aMED / MED proxy
# beneficial: fruit, veg, whole grain, fish
# detrimental: processed meat/refined grain
# moderate alcohol: 중간 섭취 구간에 1점
# =====================================================

amed_components = pd.DataFrame(index=df.index)

amed_components["fruit"] = median_score_positive(fruit)
amed_components["vegetable"] = median_score_positive(vegetable)
amed_components["whole_grain"] = median_score_positive(whole_grain)
amed_components["fish"] = median_score_positive(fish)
amed_components["meat_processed_reverse"] = median_score_negative(meat_processed)
amed_components["refined_grain_reverse"] = median_score_negative(refined_grain)

# alcohol: z-score 기반이면 중앙부를 moderate로 간주
if alcohol.notna().sum() > 0:
    lo = alcohol.quantile(0.25)
    hi = alcohol.quantile(0.75)
    amed_components["moderate_alcohol"] = np.where(
        (alcohol >= lo) & (alcohol <= hi), 1, 0
    )
else:
    amed_components["moderate_alcohol"] = np.nan

# MUFA/SFA 있으면 추가
if mufa.notna().sum() > 0 and sfa.notna().sum() > 0:
    ratio = mufa / sfa.replace(0, np.nan)
    amed_components["mufa_sfa"] = median_score_positive(ratio)

df["score_aMED_proxy"] = amed_components.sum(axis=1, skipna=True)

# =====================================================
# 2) DASH proxy
# positive: fruit, vegetable, whole grain, dairy
# negative: sodium, meat_processed, sweet beverage
# quintile 1-5 점수
# =====================================================

dash_components = pd.DataFrame(index=df.index)

dash_components["fruit"] = quintile_score_positive(fruit)
dash_components["vegetable"] = quintile_score_positive(vegetable)
dash_components["whole_grain"] = quintile_score_positive(whole_grain)
dash_components["dairy"] = quintile_score_positive(dairy)
dash_components["meat_processed_reverse"] = quintile_score_negative(meat_processed)
dash_components["sweet_beverage_reverse"] = quintile_score_negative(sweet_bev)

if sodium.notna().sum() > 0:
    dash_components["sodium_reverse"] = quintile_score_negative(sodium)
else:
    dash_components["refined_grain_reverse"] = quintile_score_negative(refined_grain)

df["score_DASH_proxy"] = dash_components.sum(axis=1, skipna=True)

# =====================================================
# 3) AHEI proxy
# 0-100 근사. 각 component 0-10.
# =====================================================

ahei_components = pd.DataFrame(index=df.index)

ahei_components["fruit"] = ten_point_positive(fruit)
ahei_components["vegetable"] = ten_point_positive(vegetable)
ahei_components["whole_grain"] = ten_point_positive(whole_grain)
ahei_components["fish"] = ten_point_positive(fish)
ahei_components["sweet_beverage_reverse"] = ten_point_negative(sweet_bev)
ahei_components["meat_processed_reverse"] = ten_point_negative(meat_processed)
ahei_components["refined_grain_reverse"] = ten_point_negative(refined_grain)

if pufa.notna().sum() > 0:
    ahei_components["pufa"] = ten_point_positive(pufa)

if sfa.notna().sum() > 0:
    ahei_components["sfa_reverse"] = ten_point_negative(sfa)

if sodium.notna().sum() > 0:
    ahei_components["sodium_reverse"] = ten_point_negative(sodium)

df["score_AHEI_proxy"] = ahei_components.mean(axis=1, skipna=True) * 10

# =====================================================
# 4) HEI proxy
# 실제 HEI는 density per 1000 kcal 기반이라 여기서는 adapted HEI.
# positive + moderation components로 구성.
# =====================================================

hei_components = pd.DataFrame(index=df.index)

hei_components["fruit"] = ten_point_positive(fruit)
hei_components["vegetable"] = ten_point_positive(vegetable)
hei_components["whole_grain"] = ten_point_positive(whole_grain)
hei_components["dairy"] = ten_point_positive(dairy)
hei_components["fish_protein"] = ten_point_positive(fish)
hei_components["refined_grain_reverse"] = ten_point_negative(refined_grain)
hei_components["sweet_beverage_reverse"] = ten_point_negative(sweet_bev)
hei_components["meat_processed_reverse"] = ten_point_negative(meat_processed)

if sodium.notna().sum() > 0:
    hei_components["sodium_reverse"] = ten_point_negative(sodium)

if sfa.notna().sum() > 0:
    hei_components["sfa_reverse"] = ten_point_negative(sfa)

df["score_HEI_proxy"] = hei_components.mean(axis=1, skipna=True) * 10

# =====================================================
# 5) RFS: Recommended Food Score proxy
# 권장 식품군을 기준 이상 섭취하면 1점
# =====================================================

rfs_components = pd.DataFrame(index=df.index)

rfs_components["fruit"] = median_score_positive(fruit)
rfs_components["vegetable"] = median_score_positive(vegetable)
rfs_components["whole_grain"] = median_score_positive(whole_grain)
rfs_components["fish"] = median_score_positive(fish)
rfs_components["dairy"] = median_score_positive(dairy)

# kimchi는 건강식품군이지만 sodium issue 때문에 RFS core에서는 제외
df["score_RFS_proxy"] = rfs_components.sum(axis=1, skipna=True)

# =====================================================
# Save
# =====================================================

score_cols = [
    "score_aMED_proxy",
    "score_DASH_proxy",
    "score_AHEI_proxy",
    "score_HEI_proxy",
    "score_RFS_proxy",
]

out_file = os.path.join(OUT_DIR, "55_guideline_diet_scores_dataset.csv")
df.to_csv(out_file, index=False, encoding="utf-8-sig")

summary = df[score_cols].describe().T
summary.to_csv(
    os.path.join(OUT_DIR, "55_guideline_score_summary.csv"),
    encoding="utf-8-sig"
)

print("\n[SCORE SUMMARY]")
print(summary)

print("\n[SAVED]")
print(out_file)