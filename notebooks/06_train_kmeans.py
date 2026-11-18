
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Project folders
project_folder = Path(__file__).resolve().parent.parent
dataset_folder = project_folder / "dataset"
models_folder = project_folder / "models"
outputs_folder = project_folder / "outputs"

outputs_folder.mkdir(exist_ok=True)

# Load prepared employee data
X_train = np.load(models_folder / "X_train.npy")
X_test = np.load(models_folder / "X_test.npy")

print("Training K-Means Clustering...")
print("Training data shape:", X_train.shape)

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Train clustering model
train_clusters = kmeans.fit_predict(X_train)

# Evaluate clustering quality
silhouette = silhouette_score(X_train, train_clusters)

print("\n========== K-MEANS RESULTS ==========")
print("Number of clusters:", 3)
print("Silhouette Score:", round(silhouette, 4))

# Display number of employees in each cluster
cluster_counts = pd.Series(train_clusters).value_counts().sort_index()

print("\nEmployees in each cluster:")
print(cluster_counts)

# Load original dataset to study cluster characteristics
file_path = list(dataset_folder.glob("*.csv"))[0]
df = pd.read_csv(file_path)

# Recreate the same training row indices
from sklearn.model_selection import train_test_split

X = df.drop(
    columns=[
        "Attrition",
        "EmployeeNumber",
        "EmployeeCount",
        "Over18",
        "StandardHours"
    ]
)

y = df["Attrition"].map({"No": 0, "Yes": 1})

X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create cluster analysis table
cluster_data = X_train_original.copy()
cluster_data["Cluster"] = train_clusters
cluster_data["Attrition"] = y_train_original.to_numpy()

print("\n========== CLUSTER ATTRITION ANALYSIS ==========")

attrition_analysis = pd.crosstab(
    cluster_data["Cluster"],
    cluster_data["Attrition"],
    normalize="index"
) * 100

attrition_analysis = attrition_analysis.rename(
    columns={0: "Stayed (%)", 1: "Left (%)"}
)

print(attrition_analysis.round(2))

# Save cluster analysis
attrition_analysis.to_csv(
    outputs_folder / "cluster_attrition_analysis.csv"
)

# Plot employee count per cluster
plt.figure(figsize=(8, 5))
cluster_counts.sort_index().plot(kind="bar")
plt.title("Number of Employees in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Employees")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(outputs_folder / "kmeans_cluster_counts.png")
plt.close()

# Plot attrition percentage by cluster
plt.figure(figsize=(8, 5))
attrition_analysis["Left (%)"].plot(kind="bar")
plt.title("Employee Attrition Percentage by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Employees Who Left (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(outputs_folder / "kmeans_attrition_by_cluster.png")
plt.close()

# Save trained model
joblib.dump(kmeans, models_folder / "kmeans_model.pkl")

print("\nK-Means training completed!")
print("Model saved as models/kmeans_model.pkl")
print("Graphs and analysis saved inside outputs folder.")
