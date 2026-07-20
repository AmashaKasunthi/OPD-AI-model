import pandas as pd

# Description
description_df = pd.read_csv("../dataset/symptom_Description.csv")
description_df["Disease"] = description_df["Disease"].str.strip()

description_map = dict(
    zip(
        description_df["Disease"],
        description_df["Description"]
    )
)

# Precautions
precaution_df = pd.read_csv("../dataset/symptom_precaution.csv")
precaution_df["Disease"] = precaution_df["Disease"].str.strip()


def get_description(disease):

    disease = disease.strip()

    return description_map.get(
        disease,
        "No description available."
    )


def get_precautions(disease):

    disease = disease.strip()

    row = precaution_df[
        precaution_df["Disease"] == disease
    ]

    if row.empty:
        return [
            "Consult a doctor.",
            "Take adequate rest.",
            "Drink plenty of water.",
            "Follow prescribed medication."
        ]

    precautions = []

    for column in [
        "Precaution_1",
        "Precaution_2",
        "Precaution_3",
        "Precaution_4"
    ]:

        value = row.iloc[0][column]

        if pd.notna(value):
            precautions.append(str(value))

    return precautions