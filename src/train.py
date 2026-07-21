import pandas as pd
import joblib
import json

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================
# Load Dataset
# =====================================
df = pd.read_csv("../dataset/dataset.csv")

print("======================================")
print("Dataset Loaded Successfully")
print("======================================")
print(df.head())

# =====================================
# Get Symptom Columns
# =====================================
symptom_columns = [col for col in df.columns if col.startswith("Symptom")]

# =====================================
# Convert Symptoms into List
# =====================================
symptom_lists = []

for _, row in df.iterrows():

    symptoms = []

    for col in symptom_columns:

        value = row[col]

        if pd.notna(value):
            symptoms.append(str(value).strip())

    symptom_lists.append(symptoms)

# =====================================
# Convert Symptoms into Binary Features
# =====================================
mlb = MultiLabelBinarizer()

X = mlb.fit_transform(symptom_lists)

y = df["Disease"]

# =====================================
# Train Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
############### RANDOM FOREST ###########################

print("\n======================================")
print("Training Random Forest...")
print("======================================")

rf_model = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

rf_model.fit(X_train, y_train)

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)

rf_precision = precision_score(
    y_test,
    rf_test_pred,
    average="weighted",
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_test_pred,
    average="weighted",
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_test_pred,
    average="weighted",
    zero_division=0
)

print(f"Training Accuracy : {rf_train_acc*100:.2f}%")
print(f"Testing Accuracy  : {rf_test_acc*100:.2f}%")
print(f"Precision         : {rf_precision*100:.2f}%")
print(f"Recall            : {rf_recall*100:.2f}%")
print(f"F1 Score          : {rf_f1*100:.2f}%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_test_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_test_pred))

################## LIGHTGBM #############################
print("\n======================================")
print("Training LightGBM...")
print("======================================")

lgbm_model = LGBMClassifier(
    n_estimators=500,
    random_state=42
)

lgbm_model.fit(X_train, y_train)

lgbm_train_pred = lgbm_model.predict(X_train)
lgbm_test_pred = lgbm_model.predict(X_test)

lgbm_train_acc = accuracy_score(y_train, lgbm_train_pred)
lgbm_test_acc = accuracy_score(y_test, lgbm_test_pred)

lgbm_precision = precision_score(
    y_test,
    lgbm_test_pred,
    average="weighted",
    zero_division=0
)

lgbm_recall = recall_score(
    y_test,
    lgbm_test_pred,
    average="weighted",
    zero_division=0
)

lgbm_f1 = f1_score(
    y_test,
    lgbm_test_pred,
    average="weighted",
    zero_division=0
)

print(f"Training Accuracy : {lgbm_train_acc*100:.2f}%")
print(f"Testing Accuracy  : {lgbm_test_acc*100:.2f}%")
print(f"Precision         : {lgbm_precision*100:.2f}%")
print(f"Recall            : {lgbm_recall*100:.2f}%")
print(f"F1 Score          : {lgbm_f1*100:.2f}%")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, lgbm_test_pred))

print("\nClassification Report")
print(classification_report(y_test, lgbm_test_pred))

################ MODEL COMPARISON ########################
print("\n======================================")
print("MODEL COMPARISON")
print("======================================")

print(f"{'Metric':<25}{'Random Forest':<20}{'LightGBM'}")
print("-"*60)

print(f"{'Training Accuracy':<25}{rf_train_acc*100:<20.2f}{lgbm_train_acc*100:.2f}")
print(f"{'Testing Accuracy':<25}{rf_test_acc*100:<20.2f}{lgbm_test_acc*100:.2f}")
print(f"{'Precision':<25}{rf_precision*100:<20.2f}{lgbm_precision*100:.2f}")
print(f"{'Recall':<25}{rf_recall*100:<20.2f}{lgbm_recall*100:.2f}")
print(f"{'F1 Score':<25}{rf_f1*100:<20.2f}{lgbm_f1*100:.2f}")

################ SAVE BEST MODEL #########################

if rf_test_acc >= lgbm_test_acc:

    best_model = rf_model
    best_name = "Random Forest"
    best_accuracy = rf_test_acc

else:

    best_model = lgbm_model
    best_name = "LightGBM"
    best_accuracy = lgbm_test_acc

joblib.dump(best_model, "../model/model.pkl")
joblib.dump(mlb, "../model/symptoms.pkl")

print("\n======================================")
print("BEST MODEL")
print("======================================")
print("Selected Model :", best_name)
print(f"Best Accuracy  : {best_accuracy*100:.2f}%")
print("Model Saved Successfully")
print("======================================")

# =====================================
# Save Model Metrics
# =====================================

metrics = {
    "best_model": best_name,
    "training_accuracy": round(
        rf_train_acc * 100 if best_name == "Random Forest" else lgbm_train_acc * 100,
        2
    ),
    "testing_accuracy": round(best_accuracy * 100, 2),
    "precision": round(
        rf_precision * 100 if best_name == "Random Forest" else lgbm_precision * 100,
        2
    ),
    "recall": round(
        rf_recall * 100 if best_name == "Random Forest" else lgbm_recall * 100,
        2
    ),
    "f1_score": round(
        rf_f1 * 100 if best_name == "Random Forest" else lgbm_f1 * 100,
        2
    )
}

with open("../model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics Saved Successfully")