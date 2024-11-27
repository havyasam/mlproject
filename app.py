# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load both models
rf_model = joblib.load("rf_model.pkl")
svm_model = joblib.load("svm_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        # Extract input features from the JSON data
        input_features = [
            data.get("Pregnancies", 0),
            data.get("Glucose", 0),
            data.get("BloodPressure", 0),
            data.get("Insulin", 0),
            data.get("BMI", 0),
            data.get("Age", 0),
        ]

        # Convert input features to a 2D array
        input_array = np.array(input_features).reshape(1, -1)

        # Make predictions with both models
        rf_probability = rf_model.predict_proba(input_array)[0][1] * 100
        svm_probability = svm_model.predict_proba(input_array)[0][1] * 100

        return jsonify({
            "RandomForest_probability": f"{rf_probability:.2f}",
            "SVM_probability": f"{svm_probability:.2f}"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
