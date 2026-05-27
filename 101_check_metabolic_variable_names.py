import pandas as pd

FILE = r"D:\precision_nutrition\FFQ\results\99_isolated_preDM_latent_foodpattern_cluster\99_isolated_preDM_cluster_assigned_dataset.csv"

df = pd.read_csv(FILE, nrows=5)

targets = [
    "glu",
    "glucose",
    "tg",
    "hdl",
    "waist",
    "sbp",
    "dbp",
    "hba",
    "a1c",
    "bmi"
]

for t in targets:
    print(f"\n===== {t} =====")

    cols = [
        c for c in df.columns
        if t.lower() in c.lower()
    ]

    print(cols)