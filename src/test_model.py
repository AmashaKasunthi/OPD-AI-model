import joblib

model = joblib.load("../model/model.pkl")
mlb = joblib.load("../model/symptoms.pkl")

symptoms = [
    "itching",
    "skin_rash",
    "nodal_skin_eruptions"
]

X = mlb.transform([symptoms])

prediction = model.predict(X)

print("Predicted Disease:", prediction[0])