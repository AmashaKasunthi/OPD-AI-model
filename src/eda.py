import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================
# Load Dataset
# =====================================
print("======================================")
print("Loading Dataset...")
print("======================================")

df = pd.read_csv("../dataset/dataset.csv")

print("Dataset Loaded Successfully!")
print()

# =====================================
# Display First 5 Records
# =====================================
print("======================================")
print("FIRST 5 RECORDS")
print("======================================")
print(df.head())

# =====================================
# Dataset Shape
# =====================================
print("\n======================================")
print("DATASET SHAPE")
print("======================================")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# =====================================
# Dataset Information
# =====================================
print("\n======================================")
print("DATASET INFORMATION")
print("======================================")
df.info()

# =====================================
# Data Types
# =====================================
print("\n======================================")
print("DATA TYPES")
print("======================================")
print(df.dtypes)

# =====================================
# Missing Values
# =====================================
print("\n======================================")
print("MISSING VALUES")
print("======================================")
print(df.isnull().sum())

# =====================================
# Duplicate Records
# =====================================
print("\n======================================")
print("DUPLICATE RECORDS")
print("======================================")

duplicate_count = df.duplicated().sum()

print(f"Duplicate Records : {duplicate_count}")

# =====================================
# Disease Distribution
# =====================================
print("\n======================================")
print("DISEASE DISTRIBUTION")
print("======================================")
print(df["Disease"].value_counts())

# =====================================
# Summary Statistics
# =====================================
print("\n======================================")
print("SUMMARY STATISTICS")
print("======================================")
print(df.describe(include="all"))

print("\n======================================")
print(" COMPLETED SUCCESSFULLY")
print("======================================")

# =====================================
# Create Graph Folder
# =====================================

graph_folder = "../graphs"

os.makedirs(graph_folder, exist_ok=True)

# =====================================
# Disease Distribution - Bar Chart
# =====================================

disease_counts = df["Disease"].value_counts()

plt.figure(figsize=(14,8))

disease_counts.plot(
    kind="bar",
    color="skyblue",
    edgecolor="black"
)

plt.title("Disease Distribution")
plt.xlabel("Disease")
plt.ylabel("Number of Patients")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(f"{graph_folder}/disease_distribution_bar.png")

plt.show()

print("Disease Distribution Bar Chart Saved.")

# =====================================
# Disease Distribution - Pie Chart
# =====================================

plt.figure(figsize=(10,10))

disease_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.ylabel("")

plt.title("Disease Distribution")

plt.tight_layout()

plt.savefig(f"{graph_folder}/disease_distribution_pie.png")

plt.show()

print("Disease Distribution Pie Chart Saved.")

# =====================================
# Symptom Frequency Analysis
# =====================================

print("\n======================================")
print("SYMPTOM FREQUENCY ANALYSIS")
print("======================================")

# Get all symptom columns
symptom_columns = [col for col in df.columns if col.startswith("Symptom")]

# Combine all symptoms into one list
all_symptoms = []

for col in symptom_columns:
    symptoms = df[col].dropna().astype(str)
    all_symptoms.extend(symptoms)

# Count frequency of each symptom
symptom_frequency = pd.Series(all_symptoms).value_counts()

print(symptom_frequency.head(20))

# Plot Top 20 Symptoms
plt.figure(figsize=(14,8))

symptom_frequency.head(20).plot(
    kind="bar",
    color="orange",
    edgecolor="black"
)

plt.title("Top 20 Most Frequent Symptoms")
plt.xlabel("Symptoms")
plt.ylabel("Frequency")

plt.xticks(rotation=60)

plt.tight_layout()

plt.savefig(f"{graph_folder}/top20_symptoms.png")

plt.show()

print("Top 20 Symptoms Chart Saved.")

# =====================================
# Missing Values Chart
# =====================================

missing_values = df.isnull().sum()

plt.figure(figsize=(12,6))

missing_values.plot(
    kind="bar",
    color="red",
    edgecolor="black"
)

plt.title("Missing Values per Column")
plt.xlabel("Columns")
plt.ylabel("Missing Values")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig(f"{graph_folder}/missing_values.png")

plt.show()

print("Missing Values Chart Saved.")

# =====================================
# Number of Symptoms Per Patient
# =====================================

symptom_count = df[symptom_columns].notna().sum(axis=1)

plt.figure(figsize=(10,6))

plt.hist(
    symptom_count,
    bins=15,
    edgecolor="black"
)

plt.title("Number of Symptoms per Patient")
plt.xlabel("Number of Symptoms")
plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig(f"{graph_folder}/symptoms_per_patient.png")

plt.show()

print("Symptoms Per Patient Histogram Saved")