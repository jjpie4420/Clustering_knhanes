import pandas as pd
from config import PROCESSED_DIR

INPUT = PROCESSED_DIR / "knhanes_ffq_rc_merged_2012_2016.csv"

KEYWORDS = [
    "DM",
    "HP",
    "HL",
    "dg",
    "dr",
    "pt",
    "med",
]

def main():

    df = pd.read_csv(INPUT, encoding="utf-8-sig", low_memory=False)

    cols = df.columns.tolist()

    found = []

    for c in cols:
        c_upper = c.upper()

        if any(k.upper() in c_upper for k in KEYWORDS):
            found.append(c)

    found = sorted(found)

    print("\n[MEDICATION / DIAGNOSIS RELATED VARIABLES]\n")

    for c in found:
        print(c)

if __name__ == "__main__":
    main()