from flask import Flask, jsonify, render_template, request
from pathlib import Path
import joblib
import tensorflow as tf
import pandas as pd

# Find the main project folder
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
MODELS_FOLDER = PROJECT_FOLDER / "models"

# Create Flask application
app = Flask(__name__)

# Load preprocessing system
preprocessor = joblib.load(
    MODELS_FOLDER / "preprocessor.pkl"
)

# Load trained models
ann_model = tf.keras.models.load_model(
    MODELS_FOLDER / "ann_model.keras"
)

svm_model = joblib.load(
    MODELS_FOLDER / "svm_model.pkl"
)

kmeans_model = joblib.load(
    MODELS_FOLDER / "kmeans_model.pkl"
)


# Homepage
@app.route("/")
def home():
    return render_template("index.html")

# Employee attrition prediction API
@app.route("/api/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "Please provide employee details."
            }), 400

        employee = data.get("employee", {})
        selected_model = data.get("model", "ANN").upper()

        # Check required employee features
        required_features = list(preprocessor.feature_names_in_)

        missing_features = [
            feature for feature in required_features
            if feature not in employee
        ]

        if missing_features:
            return jsonify({
                "error": "Missing employee information.",
                "missing_features": missing_features
            }), 400

        # Prepare employee data
        employee_df = pd.DataFrame(
            [employee],
            columns=required_features
        )

        # Apply the saved preprocessing system
        processed_data = preprocessor.transform(employee_df)

        # Select the classification model
        if selected_model == "ANN":
            probability = float(
                ann_model.predict(
                    processed_data, verbose=0
                )[0][0]
            )

        elif selected_model == "SVM":
            probability = float(
                svm_model.predict_proba(
                    processed_data
                )[0][1]
            )

        else:
            return jsonify({
                "error": "Select either ANN or SVM."
            }), 400

        prediction = (
            "Likely to Leave"
            if probability >= 0.5
            else "Likely to Stay"
        )

        # Identify the employee's K-Means cluster
        cluster = int(
            kmeans_model.predict(processed_data)[0]
        )

        return jsonify({
            "status": "success",
            "model": selected_model,
            "prediction": prediction,
            "attrition_probability": round(
                probability * 100, 2
            ),
            "cluster": cluster
        })

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


# Backend health check
@app.route("/api/health")
def health():
    return jsonify({
        "status": "success",
        "message": "Backend is running successfully!",
        "models_loaded": [
            "ANN",
            "SVM",
            "K-Means"
        ],
        "preprocessor": "Loaded successfully"
    })


# Model comparison API
@app.route("/api/comparison")
def model_comparison():

    comparison_path = (
        PROJECT_FOLDER
        / "outputs"
        / "model_comparison.csv"
    )

    results = pd.read_csv(comparison_path)

    return jsonify(
        results.to_dict(orient="records")
    )

@app.route("/api/clusters")
def cluster_analysis():
    try:
        output_folder = PROJECT_FOLDER / "outputs"

        employee_profiles = pd.read_csv(
            output_folder / "cluster_employee_profiles.csv"
        )

        overtime_profiles = pd.read_csv(
            output_folder / "cluster_overtime_profiles.csv"
        )

        attrition_profiles = pd.read_csv(
            output_folder / "cluster_attrition_profiles.csv"
        )

        return jsonify({
            "status": "success",
            "employee_profiles": employee_profiles.to_dict(orient="records"),
            "overtime_profiles": overtime_profiles.to_dict(orient="records"),
            "attrition_profiles": attrition_profiles.to_dict(orient="records")
        })

    except Exception as error:
        return jsonify({"error": str(error)}), 500

if __name__ == "__main__":
    print("Loading Employee Attrition models...")
    print("ANN model loaded!")
    print("SVM model loaded!")
    print("K-Means model loaded!")
    print("Preprocessor loaded!")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
