
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Project paths
project_folder = Path(__file__).resolve().parent.parent
models_folder = project_folder / "models"
outputs_folder = project_folder / "outputs"
outputs_folder.mkdir(exist_ok=True)

# Load prepared data
X_train = np.load(models_folder / "X_train.npy")
X_test = np.load(models_folder / "X_test.npy")
y_train = np.load(models_folder / "y_train.npy")
y_test = np.load(models_folder / "y_test.npy")

print("Training ANN model...")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Calculate class weights because the dataset is imbalanced
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes, weights))

# Create ANN architecture
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

# Stop training if the model stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# Predict employee attrition
probabilities = model.predict(X_test).ravel()
predictions = (probabilities >= 0.5).astype(int)

# Evaluate model
print("\n========== ANN MODEL RESULTS ==========")
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
model.save(models_folder / "ann_model.keras")

# Save training history
joblib.dump(history.history, models_folder / "ann_history.pkl")

# Save confusion matrix graph
cm = confusion_matrix(y_test, predictions)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Left"]
)
display.plot(cmap="Blues")
plt.title("ANN Confusion Matrix")
plt.tight_layout()
plt.savefig(outputs_folder / "ann_confusion_matrix.png")
plt.close()

# Save training accuracy graph
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("ANN Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(outputs_folder / "ann_training_accuracy.png")
plt.close()

print("\nANN training completed!")
print("Model saved as models/ann_model.keras")
print("Graphs saved inside outputs folder.")
