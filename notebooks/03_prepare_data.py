
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Project paths
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"
models_folder = project_folder / "models"
models_folder.mkdir(exist_ok=True)

# Load dataset
file_path = list(dataset_folder.glob("*.csv"))[0]
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# Remove unnecessary columns
columns_to_drop = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=columns_to_drop)

# Separate input features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"].map({"No": 0, "Yes": 1})

# Identify numerical and categorical columns
numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:", len(numerical_columns))
print("Categorical columns:", len(categorical_columns))

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Prepare numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ), categorical_columns)
    ]
)

# Learn preprocessing from training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save processed data
joblib.dump(preprocessor, models_folder / "preprocessor.pkl")

np.save(models_folder / "X_train.npy", X_train_processed)
np.save(models_folder / "X_test.npy", X_test_processed)
np.save(models_folder / "y_train.npy", y_train.to_numpy())
np.save(models_folder / "y_test.npy", y_test.to_numpy())

# Save transformed feature names
feature_names = preprocessor.get_feature_names_out()
pd.Series(feature_names).to_csv(
    models_folder / "feature_names.csv",
    index=False,
    header=False
)

print("\n========== DATA PREPARATION COMPLETE ==========")
print("Training records:", X_train_processed.shape[0])
print("Testing records:", X_test_processed.shape[0])
print("Processed training features:", X_train_processed.shape[1])
print("Employees who left in training data:", int(y_train.sum()))
print("Employees who stayed in training data:", int((y_train == 0).sum()))

print("\nAll processed files saved inside the models folder!")
