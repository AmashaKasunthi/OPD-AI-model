import pandas as pd

severity_df = pd.read_csv("../dataset/Symptom-severity.csv")

severity_map = dict(
    zip(
        severity_df["Symptom"].str.strip(),
        severity_df["weight"]
    )
)

def calculate_severity(symptoms):

    total_score = 0

    for symptom in symptoms:
        total_score += severity_map.get(symptom.strip(), 0)

    return total_score


def get_risk_level(score):

    if score < 10:
        return "Low"

    elif score < 20:
        return "Medium"

    else:
        return "High"