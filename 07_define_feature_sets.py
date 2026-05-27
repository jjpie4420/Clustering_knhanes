import json
import pandas as pd
from config import PROCESSED_DIR, RESULT_DIR

INPUT = PROCESSED_DIR / "analysis_cohort_2012_2016.csv"
OUTPUT_JSON = RESULT_DIR / "07_feature_sets.json"
OUTPUT_CSV = RESULT_DIR / "07_feature_set_summary.csv"

TARGETS = {
    "diabetes": "label_diabetes",
    "prediabetes": "label_prediabetes",
    "dyslipidemia": "label_dyslipidemia",
    "hypertension": "label_hypertension",
    "prehypertension": "label_prehypertension",
    "obesity": "label_obesity",
}

BASE_CLINICAL = [
    "age",
    "sex",
    "incm",
    "edu",
    "sm_presnt",
    "dr_month",
    "pa_aerobic",
    "HE_BMI",
]

TARGET_EXCLUDE_CLINICAL = {
    "diabetes": [],
    "prediabetes": [],
    "dyslipidemia": [],
    "hypertension": [],
    "prehypertension": [],
    "obesity": ["HE_BMI"],
}

PAIR_NUTRIENTS = {
    "EN": ("FQ_EN", "RC_EN"),
    "PROT": ("FQ_PROT", "RC_PROT"),
    "FAT": ("FQ_FAT", "RC_FAT"),
    "SFA": ("FQ_SFA", "RC_SFA"),
    "MUFA": ("FQ_MUFA", "RC_MUFA"),
    "PUFA": ("FQ_PUFA", "RC_PUFA"),
    "N3": ("FQ_N3", "RC_N3"),
    "N6": ("FQ_N6", "RC_N6"),
    "CHOL": ("FQ_CHOL", "RC_CHOL"),
    "CHO": ("FQ_CHO", "RC_CHO"),
    "TDF": ("FQ_TDF", "RC_TDF"),
    "CA": ("FQ_CA", "RC_CA"),
    "PHOS": ("FQ_PHOS", "RC_PHOS"),
    "FE": ("FQ_FE", "RC_FE"),
    "NA": ("FQ_NA", "RC_NA"),
    "K": ("FQ_K", "RC_K"),
    "VA": ("FQ_VA", "RC_VA"),
    "CAROT": ("FQ_CAROT", "RC_CAROT"),
    "RETIN": ("FQ_RETIN", "RC_RETIN"),
    "B1": ("FQ_B1", "RC_B1"),
    "B2": ("FQ_B2", "RC_B2"),
    "NIAC": ("FQ_NIAC", "RC_NIAC"),
    "VITC": ("FQ_VITC", "RC_VITC"),
}

RC_BEHAVIOR = [
    "rc_food_item_count",
    "rc_meal_count",
    "rc_food_group_count",
]

def existing(cols, df):
    return [c for c in cols if c in df.columns]

def main():
    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    feature_sets = {}

    ffq_nutrients = []
    rc_nutrients = []

    for nutrient, (fq, rc) in PAIR_NUTRIENTS.items():
        if fq in df.columns and rc in df.columns:
            ffq_nutrients.append(fq)
            rc_nutrients.append(rc)

    ffq_food_freq = [
        c for c in df.columns
        if c.startswith("FQ_") and c not in ffq_nutrients
    ]

    rc_behavior = existing(RC_BEHAVIOR, df)

    for target_name, label_col in TARGETS.items():
        clinical = existing(BASE_CLINICAL, df)
        clinical = [
            c for c in clinical
            if c not in TARGET_EXCLUDE_CLINICAL.get(target_name, [])
        ]

        feature_sets[target_name] = {
            "label": label_col,
            "clinical": clinical,
            "ffq_nutrient": ffq_nutrients,
            "rc_nutrient": rc_nutrients,
            "ffq_food_freq": ffq_food_freq,
            "rc_behavior": rc_behavior,
            "clinical_ffq_nutrient": clinical + ffq_nutrients,
            "clinical_rc_nutrient": clinical + rc_nutrients,
            "clinical_ffq_food": clinical + ffq_food_freq,
            "clinical_rc_behavior": clinical + rc_behavior,
            "clinical_ffq_rc_nutrient": clinical + ffq_nutrients + rc_nutrients,
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(feature_sets, f, ensure_ascii=False, indent=2)

    rows = []
    for target, sets in feature_sets.items():
        for set_name, cols in sets.items():
            if set_name == "label":
                continue
            rows.append({
                "target": target,
                "feature_set": set_name,
                "n_features": len(cols),
                "features": ",".join(cols)
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(summary[["target", "feature_set", "n_features"]])
    print(f"\n[SAVED] {OUTPUT_JSON}")
    print(f"[SAVED] {OUTPUT_CSV}")

if __name__ == "__main__":
    main()