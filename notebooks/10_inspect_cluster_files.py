import pandas as pd
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"

files = [
    "cluster_employee_profiles.csv",
    "cluster_overtime_profiles.csv",
    "cluster_attrition_profiles.csv"
]

for filename in files:
    print("\n" + "=" * 60)
    print(filename)

    df = pd.read_csv(OUTPUT_FOLDER / filename)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData:")
    print(df.to_string(index=False))