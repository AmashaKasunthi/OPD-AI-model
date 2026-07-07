from flask import Flask, request, jsonify
import joblib

from severity import calculate_severity, get_risk_level
from disease_info import (
    get_description,
    get_precautions
)

app = Flask(__name__)

model = joblib.load("../model/model.pkl")
mlb = joblib.load("../model/symptoms.pkl")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    symptoms = data["symptoms"]

    X = mlb.transform([symptoms])

    disease = model.predict(X)[0]

    score = calculate_severity(symptoms)

    risk_level = get_risk_level(score)

    description = get_description(disease)

    precautions = get_precautions(disease)

    return jsonify({
        "predictedDisease": disease,
        "severityScore": score,
        "riskLevel": risk_level,
        "description": description,
        "precautions": precautions
    })


if __name__ == "__main__":
    app.run(debug=True)