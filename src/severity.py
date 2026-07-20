import pandas as pd

# Load symptom severity file
severity_df = pd.read_csv("../dataset/Symptom-severity.csv")

# Remove spaces from symptom names
severity_df["Symptom"] = severity_df["Symptom"].str.strip()

# Create dictionary
severity_map = dict(zip(
    severity_df["Symptom"],
    severity_df["weight"]
))


def calculate_severity(symptoms):

    score = 0

    # If symptoms come as a string
    if isinstance(symptoms, str):
        symptoms = symptoms.split(",")

    for symptom in symptoms:

        symptom = symptom.strip()

        score += severity_map.get(symptom, 0)

    return score


def get_risk_level(score):

    if score <= 6:
        return "Low"

    elif score <= 12:
        return "Medium"

    else:
        return "High"