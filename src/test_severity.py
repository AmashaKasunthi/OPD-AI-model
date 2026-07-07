from severity import calculate_severity, get_risk_level

symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions"
]

score = calculate_severity(symptoms)

risk = get_risk_level(score)

print("Severity Score:", score)
print("Risk Level:", risk)