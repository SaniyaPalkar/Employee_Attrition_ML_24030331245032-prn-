
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# Project folders
project_folder = Path(__file__).resolve().parent.parent
models_folder = project_folder / "models"
outputs_folder = project_folder / "outputs"
outputs_folder.mkdir(exist_ok=True)

# Load prepared data
X_train = np.load(models_folder / "X_train.npy")
X_test = np.load(models_folder / "X_test.npy")
y_train = np.load(models_folder / "y_train.npy")
y_test = np.load(models_folder / "y_test.npy")

print("Training SVM model...")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Handle class imbalance
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

# Create SVM model
model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    class_weight=class_weights,
    probability=True,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# Evaluate model
print("\n========== SVM MODEL RESULTS ==========")

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, predictions, zero_division=0), 4))
print("ROC AUC:", round(roc_auc_score(y_test, probabilities), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Stayed", "Left"],
    zero_division=0
))

# Save trained model
joblib.dump(model, models_folder / "svm_model.pkl")

# Save confusion matrix
cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Left"]
)

display.plot(cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig(outputs_folder / "svm_confusion_matrix.png")
plt.close()

print("\nSVM training completed!")
print("Model saved as models/svm_model.pkl")
print("Confusion matrix saved inside outputs folder.")
