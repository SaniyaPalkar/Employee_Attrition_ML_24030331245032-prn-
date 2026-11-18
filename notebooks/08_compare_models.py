
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Project folders
project_folder = Path(__file__).resolve().parent.parent
models_folder = project_folder / "models"
outputs_folder = project_folder / "outputs"

outputs_folder.mkdir(exist_ok=True)

# Load testing data
X_test = np.load(models_folder / "X_test.npy")
y_test = np.load(models_folder / "y_test.npy")

# Load trained models
ann_model = tf.keras.models.load_model(
    models_folder / "ann_model.keras"
)

svm_model = joblib.load(
    models_folder / "svm_model.pkl"
)

# ANN predictions
ann_probabilities = ann_model.predict(
    X_test, verbose=0
).ravel()

ann_predictions = (
    ann_probabilities >= 0.5
).astype(int)

# SVM predictions
svm_probabilities = svm_model.predict_proba(
    X_test
)[:, 1]

svm_predictions = (
    svm_probabilities >= 0.5
).astype(int)

# Calculate evaluation metrics
def evaluate_model(name, predictions, probabilities):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(
            y_test, probabilities
        )
    }

results = pd.DataFrame([
    evaluate_model(
        "ANN",
        ann_predictions,
        ann_probabilities
    ),
    evaluate_model(
        "SVM",
        svm_predictions,
        svm_probabilities
    )
])

# Display results
print("\n========== ANN vs SVM COMPARISON ==========")
print(results.round(4).to_string(index=False))

# Save comparison table
results.to_csv(
    outputs_folder / "model_comparison.csv",
    index=False
)

# Generate accuracy comparison graph
plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["Accuracy"] * 100
)

plt.title("ANN vs SVM - Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for index, value in enumerate(results["Accuracy"]):
    plt.text(
        index,
        value * 100 + 1,
        f"{value * 100:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    outputs_folder / "model_accuracy_comparison.png"
)

plt.close()

print("\nModel comparison completed!")
print("Comparison table saved in outputs/model_comparison.csv")
print("Graph saved in outputs/model_accuracy_comparison.png")
