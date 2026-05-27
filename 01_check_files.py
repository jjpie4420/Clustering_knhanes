import pandas as pd
from config import DATA_DIR, FILE_MAP, YEARS, RESULT_DIR

def read_csv_safely(path, nrows=None):
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Cannot decode {path}")

def main():
    rows = []

    for year in YEARS:
        for dtype, fname in FILE_MAP[year].items():
            path = DATA_DIR / fname

            if not path.exists():
                rows.append({
                    "year": year,
                    "type": dtype,
                    "file": fname,
                    "exists": False,
                    "n_cols": None,
                    "n_rows_sample": None
                })
                continue

            df_head = read_csv_safely(path, nrows=5)
            cols = list(df_head.columns)

            rows.append({
                "year": year,
                "type": dtype,
                "file": fname,
                "exists": True,
                "n_cols": len(cols),
                "n_rows_sample": len(df_head)
            })

    out = pd.DataFrame(rows)
    print(out)

    out.to_csv(RESULT_DIR / "01_file_check.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()