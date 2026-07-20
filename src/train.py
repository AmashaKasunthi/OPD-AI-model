import pandas as pd
import joblib

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load dataset
df = pd.read_csv("../dataset/dataset.csv")

print("Dataset Loaded")
print(df.head())

# Get symptom columns
symptom_columns = [col for col in df.columns if col.startswith("Symptom")]

# Convert rows to symptom lists
symptom_lists = []

for _, row in df.iterrows():

    symptoms = []

    for col in symptom_columns:

        value = row[col]

        if pd.notna(value):
            symptoms.append(str(value).strip())

    symptom_lists.append(symptoms)

# Convert symptoms into binary features
mlb = MultiLabelBinarizer()

X = mlb.fit_transform(symptom_lists)

y = df["Disease"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

model.fit(X_train, y_train)

# Accuracy
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "../model/model.pkl")
joblib.dump(mlb, "../model/symptoms.pkl")

print("Model Saved Successfully")