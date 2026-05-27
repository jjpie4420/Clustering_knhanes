import pandas as pd
from config import DATA_DIR, FILE_MAP, YEARS, PROCESSED_DIR

ID_COL_CANDIDATES = ["ID", "id"]

BASE_COLS = [
    "ID", "year", "sex", "age", "region", "town_t", "psu", "kstrata"
]

NUTRIENT_COLS = [
    "NF_EN", "NF_WATER", "NF_PROT", "NF_FAT", "NF_SFA", "NF_MUFA", "NF_PUFA",
    "NF_N3", "NF_N6", "NF_CHOL", "NF_chol", "NF_CHO", "NF_TDF", "NF_tdf",
    "NF_SUGAR", "NF_CA", "NF_PHOS", "NF_NA", "NF_K", "NF_MG", "NF_FE", "NF_ZN",
    "NF_VA", "NF_VA_RAE", "NF_VITD", "NF_VITE", "NF_CAROT", "NF_RETIN",
    "NF_B1", "NF_B2", "NF_NIAC", "NF_FOLATE", "NF_VITC"
]

FOOD_COLS = [
    "N_FCODE", "N_FNAME", "N_KINDG1", "N_KINDG2", "N_MEAL", "N_MTYPE"
]

def read_csv_safely(path):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode file: {path}")

def normalize_columns(df):
    df = df.copy()

    if "id" in df.columns and "ID" not in df.columns:
        df = df.rename(columns={"id": "ID"})

    rename_map = {
        "NF_chol": "NF_CHOL",
        "NF_tdf": "NF_TDF",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df

def build_person_level_rc(df):
    df = normalize_columns(df)

    if "ID" not in df.columns:
        raise ValueError("ID column not found in recall data")

    if "year" not in df.columns:
        raise ValueError("year column not found in recall data")

    nutrient_cols = [c for c in NUTRIENT_COLS if c in df.columns]
    nutrient_cols = sorted(set(nutrient_cols))

    for c in nutrient_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    base_cols = [c for c in BASE_COLS if c in df.columns]

    nutrient_agg = df.groupby(["ID", "year"], as_index=False)[nutrient_cols].sum(min_count=1)

    count_features = df.groupby(["ID", "year"], as_index=False).agg(
        rc_food_item_count=("N_FNAME", "count") if "N_FNAME" in df.columns else ("ID", "count"),
        rc_meal_count=("N_MEAL", "nunique") if "N_MEAL" in df.columns else ("ID", "count"),
    )

    if "N_KINDG1" in df.columns:
        diversity = df.groupby(["ID", "year"], as_index=False).agg(
            rc_food_group_count=("N_KINDG1", "nunique")
        )
    else:
        diversity = df.groupby(["ID", "year"], as_index=False).agg(
            rc_food_group_count=("ID", "count")
        )
        diversity["rc_food_group_count"] = pd.NA

    base = df[base_cols].drop_duplicates(subset=["ID", "year"])

    out = base.merge(nutrient_agg, on=["ID", "year"], how="left")
    out = out.merge(count_features, on=["ID", "year"], how="left")
    out = out.merge(diversity, on=["ID", "year"], how="left")

    rc_cols = [c for c in out.columns if c.startswith("NF_")]
    out = out.rename(columns={c: f"RC_{c.replace('NF_', '')}" for c in rc_cols})

    return out

def main():
    all_years = []

    for year in YEARS:
        path = DATA_DIR / FILE_MAP[year]["rc"]
        print(f"[READ] {path}")

        df = read_csv_safely(path)
        df = normalize_columns(df)

        if "year" not in df.columns:
            df["year"] = year

        person = build_person_level_rc(df)
        print(year, person.shape)

        all_years.append(person)

    rc_all = pd.concat(all_years, axis=0, ignore_index=True)
    rc_all.to_csv(PROCESSED_DIR / "rc_person_2012_2016.csv", index=False, encoding="utf-8-sig")

    print("[SAVED] rc_person_2012_2016.csv")
    print(rc_all.shape)

if __name__ == "__main__":
    main()