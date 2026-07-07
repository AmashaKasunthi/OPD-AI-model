import pandas as pd


description_df = pd.read_csv("../dataset/symptom_Description.csv")

description_map = dict(
    zip(
        description_df["Disease"],
        description_df["Description"]
    )
)

# Precaution file
precaution_df = pd.read_csv("../dataset/symptom_precaution.csv")


def get_description(disease):

    return description_map.get(
        disease,
        "No description available"
    )


def get_precautions(disease):

    row = precaution_df[
        precaution_df["Disease"] == disease
    ]

    if row.empty:
        return []

    precautions = []

    for col in row.columns:

        if col.startswith("Precaution"):

            value = row.iloc[0][col]

            if pd.notna(value):
                precautions.append(str(value))

    return precautions