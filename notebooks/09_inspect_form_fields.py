
import pandas as pd
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_FOLDER
    / "dataset"
    / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

df = pd.read_csv(DATASET_PATH)

categorical_columns = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]

ignored_columns = [
    "Attrition",
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

features = [
    column for column in df.columns
    if column not in ignored_columns
]

print("\n========== EMPLOYEE INPUT FIELDS ==========")

for column in features:

    print(f"\n{column}")
    print("Data type:", df[column].dtype)

    if column in categorical_columns:
        print(
            "Dropdown options:",
            sorted(df[column].dropna().unique().tolist())
        )

print("\nTotal input features:", len(features))
print("\nInspection completed!")
