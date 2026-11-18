
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate and load dataset
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"
file_path = list(dataset_folder.glob("*.csv"))[0]

df = pd.read_csv(file_path)

# Create output folder for graphs
output_folder = project_folder / "outputs"
output_folder.mkdir(exist_ok=True)

# 1. Attrition by overtime
overtime = pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition percentage by overtime:")
print(overtime.round(2))

overtime.plot(kind="bar", figsize=(7, 5))
plt.title("Employee Attrition by Overtime")
plt.xlabel("Overtime")
plt.ylabel("Percentage of Employees")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_folder / "overtime_attrition.png")
plt.close()

# 2. Attrition by department
department = pd.crosstab(
    df["Department"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition percentage by department:")
print(department.round(2))

department.plot(kind="bar", figsize=(8, 5))
plt.title("Employee Attrition by Department")
plt.xlabel("Department")
plt.ylabel("Percentage of Employees")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_folder / "department_attrition.png")
plt.close()

# 3. Attrition by job satisfaction
satisfaction = pd.crosstab(
    df["JobSatisfaction"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nAttrition percentage by job satisfaction:")
print(satisfaction.round(2))

satisfaction.plot(kind="bar", figsize=(8, 5))
plt.title("Employee Attrition by Job Satisfaction")
plt.xlabel("Job Satisfaction (1 = Low, 4 = High)")
plt.ylabel("Percentage of Employees")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(output_folder / "satisfaction_attrition.png")
plt.close()

print("\nAnalysis completed!")
print("Three graphs saved inside the outputs folder.")
