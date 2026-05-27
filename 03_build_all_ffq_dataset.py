import pandas as pd
from config import DATA_DIR, FILE_MAP, YEARS, PROCESSED_DIR

KEYS = ["ID", "year"]

CLINICAL_COLS = [
    "ID", "year",
    "sex", "age", "region", "town_t", "psu", "kstrata",
    "wt_tot", "wt_ntr", "wt_hs", "wt_itvex",
    "incm", "ho_incm", "edu", "educ", "occp",

    # fasting / disease survey
    "HE_fst",

    # diagnosis / treatment variables
    "HE_HPdg", "HE_HPdr",
    "HE_DMdg", "HE_DMdr",
    "HE_HLdg", "HE_HLdr",

    # questionnaire diagnosis / treatment
    "DI1_dg", "DI1_pr", "DI1_pt",
    "DE1_dg", "DE1_pr", "DE1_pt",
    "DI2_dg", "DI2_pr", "DI2_pt",

    # blood pressure / anthropometry
    "HE_sbp", "HE_dbp", "HE_HP",
    "HE_ht", "HE_wt", "HE_wc", "HE_BMI", "HE_obe",

    # glucose / diabetes
    "HE_glu", "HE_HbA1c", "HE_insulin", "HE_DM",

    # lipid
    "HE_chol", "HE_HDL_st2", "HE_TG", "HE_LDL_drct",
    "HE_HCHOL", "HE_hCHOL", "HE_LHDL_st2", "HE_HTG", "HE_hTG",

    # liver / kidney / inflammation
    "HE_ast", "HE_alt", "HE_crea", "HE_hsCRP",

    # lifestyle
    "sm_presnt", "dr_month", "AUDIT",
    "pa_high", "pa_mid", "pa_walk", "pa_aerobic",

    # 24h recall summary in all file
    "N_EN", "N_PROT", "N_FAT", "N_CHO", "N_TDF", "N_tdf",
    "N_NA", "N_K", "N_SUGAR",

    # diet quality
    "HEI", "HEI_BR", "HEI_CEREAL", "HEI_TFRUIT", "HEI_FFRUIT",
    "HEI_TVEG", "HEI_VEG", "HEI_PROTF", "HEI_DAIRY",
    "HEI_SFA", "HEI_NA", "HEI_SWEET", "HEI_CHO", "HEI_FAT", "HEI_EN",

    # pregnancy
    "HE_prg"
]

FFQ_PREFIXES = ["FQ_", "FF_", "FA_", "FS_"]

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
        "HE_hCHOL": "HE_HCHOL",
        "HE_hTG": "HE_HTG",
        "N_tdf": "N_TDF",
        "N_chol": "N_CHOL",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

def select_existing(df, cols):
    return [c for c in cols if c in df.columns]

def load_all_years():
    frames = []

    for year in YEARS:
        path = DATA_DIR / FILE_MAP[year]["all"]
        print(f"[READ ALL] {path}")
        df = read_csv_safely(path)
        df = normalize_columns(df)

        if "year" not in df.columns:
            df["year"] = year

        cols = select_existing(df, CLINICAL_COLS)
        df = df[cols].drop_duplicates(subset=KEYS)

        frames.append(df)

    return pd.concat(frames, axis=0, ignore_index=True)

def load_ffq_years():
    frames = []

    for year in YEARS:
        path = DATA_DIR / FILE_MAP[year]["ffq"]
        print(f"[READ FFQ] {path}")
        df = read_csv_safely(path)
        df = normalize_columns(df)

        if "year" not in df.columns:
            df["year"] = year

        ffq_cols = [c for c in df.columns if any(c.startswith(p) for p in FFQ_PREFIXES)]
        keep = KEYS + ffq_cols
        keep = [c for c in keep if c in df.columns]

        df = df[keep].drop_duplicates(subset=KEYS)
        frames.append(df)

    return pd.concat(frames, axis=0, ignore_index=True)

def main():
    all_df = load_all_years()
    ffq_df = load_ffq_years()

    rc_path = PROCESSED_DIR / "rc_person_2012_2016.csv"
    if not rc_path.exists():
        raise FileNotFoundError("먼저 02_build_recall_person_level.py 실행해야 함")

    rc_df = pd.read_csv(rc_path, encoding="utf-8-sig", low_memory=False)
    rc_df = normalize_columns(rc_df)

    print("[SHAPE] all:", all_df.shape)
    print("[SHAPE] ffq:", ffq_df.shape)
    print("[SHAPE] rc:", rc_df.shape)

    df = all_df.merge(ffq_df, on=KEYS, how="inner")
    df = df.merge(rc_df, on=KEYS, how="inner", suffixes=("", "_rcmeta"))

    print("[SHAPE] merged:", df.shape)

    df.to_csv(PROCESSED_DIR / "knhanes_ffq_rc_merged_2012_2016.csv", index=False, encoding="utf-8-sig")

    print("[SAVED] knhanes_ffq_rc_merged_2012_2016.csv")

if __name__ == "__main__":
    main()