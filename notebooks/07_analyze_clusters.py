
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split

# Project folders
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"
models_folder = project_folder / "models"
outputs_folder = project_folder / "outputs"

# Load original dataset
file_path = list(dataset_folder.glob("*.csv"))[0]
df = pd.read_csv(file_path)

# Separate features and target
X = df.drop(
    columns=[
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ]
)

y = df["Attrition"]

# Recreate the same training split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained K-Means model
import joblib

kmeans = joblib.load(models_folder / "kmeans_model.pkl")

# Load preprocessor and transform training data
preprocessor = joblib.load(models_folder / "preprocessor.pkl")
X_train_processed = preprocessor.transform(X_train)

# Predict cluster numbers
clusters = kmeans.predict(X_train_processed)

# Create analysis dataframe
cluster_df = X_train.copy()
cluster_df["Cluster"] = clusters
cluster_df["Attrition"] = y_train.to_numpy()

print("\n========== EMPLOYEE CLUSTER PROFILES ==========")

# Numerical profile
numerical_profile = cluster_df.groupby("Cluster")[
    [
        "Age",
        "MonthlyIncome",
        "JobSatisfaction",
        "YearsAtCompany",
        "TotalWorkingYears"
    ]
].mean()

print("\nAverage employee characteristics:")
print(numerical_profile.round(2))

# Overtime distribution
print("\nOvertime distribution by cluster (%):")

overtime_profile = pd.crosstab(
    cluster_df["Cluster"],
    cluster_df["OverTime"],
    normalize="index"
) * 100

print(overtime_profile.round(2))

# Attrition percentage
print("\nAttrition percentage by cluster (%):")

attrition_profile = pd.crosstab(
    cluster_df["Cluster"],
    cluster_df["Attrition"],
    normalize="index"
) * 100

attrition_profile = attrition_profile.rename(
    columns={"No": "Stayed (%)", "Yes": "Left (%)"}
)

print(attrition_profile.round(2))

# Save profile tables
numerical_profile.to_csv(
    outputs_folder / "cluster_employee_profiles.csv"
)

overtime_profile.to_csv(
    outputs_folder / "cluster_overtime_profiles.csv"
)

attrition_profile.to_csv(
    outputs_folder / "cluster_attrition_profiles.csv"
)

# Plot average monthly income
plt.figure(figsize=(8, 5))
numerical_profile["MonthlyIncome"].plot(kind="bar")
plt.title("Average Monthly Income by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Monthly Income")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(outputs_folder / "cluster_monthly_income.png")
plt.close()

# Plot average job satisfaction
plt.figure(figsize=(8, 5))
numerical_profile["JobSatisfaction"].plot(kind="bar")
plt.title("Average Job Satisfaction by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Job Satisfaction")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(outputs_folder / "cluster_job_satisfaction.png")
plt.close()

# Plot average years at company
plt.figure(figsize=(8, 5))
numerical_profile["YearsAtCompany"].plot(kind="bar")
plt.title("Average Years at Company by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Years")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(outputs_folder / "cluster_years_at_company.png")
plt.close()

print("\nCluster analysis completed!")
print("Profile tables and graphs saved inside outputs folder.")
