from flask import Flask, request, jsonify
import joblib
import re
from flask import Flask, request, jsonify

from severity import calculate_severity, get_risk_level
from disease_info import get_description, get_precautions

app = Flask(__name__)

# Load trained model
model = joblib.load("../model/model.pkl")
mlb = joblib.load("../model/symptoms.pkl")

import json

with open("../model/metrics.json", "r") as f:
    metrics = json.load(f)

print("\n========== MODEL INFORMATION ==========")
print("Best Model       :", metrics["best_model"])
print("Training Accuracy:", metrics["training_accuracy"], "%")
print("Testing Accuracy :", metrics["testing_accuracy"], "%")
print("Precision        :", metrics["precision"], "%")
print("Recall           :", metrics["recall"], "%")
print("F1 Score         :", metrics["f1_score"], "%")
print("=======================================\n")

@app.route("/model_metrics", methods=["GET"])
def model_metrics():
    return jsonify(metrics)


@app.route("/predict", methods=["POST"])
def predict():
    try:

        # Receive JSON
        data = request.get_json()

        print("\n========== NEW REQUEST ==========")
        print("Received JSON:")
        print(data)

        # Get symptoms
        symptoms = data.get("symptoms", "")

        print("\nRaw Symptoms:")
        print(symptoms)

        # Convert string into list
        if isinstance(symptoms, str):

            symptom_list = [
        s.strip().lower().replace(" ", "_")
        for s in re.split(r"[,\n]+", symptoms)
        if s.strip()
    ]

        else:

            symptom_list = [
                str(s).strip().lower().replace(" ", "_")
                for s in symptoms
            ]

        print("\nConverted Symptom List:")
        print(symptom_list)

        # Convert symptoms into model input
        X = mlb.transform([symptom_list])

        print("\nEncoded Input:")
        print(X)

        # Disease prediction
        disease = model.predict(X)[0]

        # Severity score
        score = calculate_severity(symptom_list)

        # Risk level
        risk_level = get_risk_level(score)

        # Description
        description = get_description(disease)

        # Precautions
        precautions = get_precautions(disease)

        print("\n========== AI RESULT ==========")
        print("Disease :", disease)
        print("Severity:", score)
        print("Risk    :", risk_level)
        print("Description:", description)
        print("Precautions:", precautions)
        print("===============================\n")

        return jsonify({
            "predictedDisease": disease,
            "severityScore": score,
            "riskLevel": risk_level,
            "description": description,
            "precautions": precautions
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/")
def home():
    return "OPD AI Prediction API Running Successfully"


if __name__ == "__main__":
    app.run(debug=True)

