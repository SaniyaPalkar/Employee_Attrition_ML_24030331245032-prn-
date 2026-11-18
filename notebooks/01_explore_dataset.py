
import pandas as pd
from pathlib import Path

# Find the dataset
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"

csv_files = list(dataset_folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found inside the dataset folder.")

file_path = csv_files[0]

# Load employee data
df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")

# Number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display first five records
print("\nFirst five employee records:")
print(df.head())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check employee attrition
print("\nAttrition distribution:")
print(df["Attrition"].value_counts())

print("\nDataset loaded successfully!")

