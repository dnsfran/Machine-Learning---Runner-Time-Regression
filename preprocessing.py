import pandas as pd
import re

# Load the dataset
df = pd.read_csv("machine learning\TWO_CENTURIES_OF_UM_RACES.csv")

# Runs in 2020
df["Year of event"] = pd.to_numeric(df["Year of event"], errors="coerce")
df_2020 = df[df["Year of event"] == 2020]


def convert_to_km(distance_str):
    distance_str = str(distance_str).strip().lower()
    # Recherche du nombre
    match = re.search(r'(\d+(\.\d+)?)', distance_str)
    if not match:
        return None
    value = float(match.group(1))
    # Si la distance est en miles, convertir
    if "mi" in distance_str:

        value *= 1.60934
    return value

# Filter runs that are lenghts to complete and not the longest you can make in a limited time
mask_fixed_distance = df_2020["Event distance/length"].str.contains(r'^\d+(\.\d+)?\s*(km|mi)s?$', regex=True, case=False) 
#tries to look if the event is in miles or kilometers otherwhise it is in a limited time
df_fixed = df_2020[mask_fixed_distance].copy()

# Convert all lenghts to km
df_fixed["Distance_km"] = df_fixed["Event distance/length"].apply(convert_to_km)

def remove_top_5_percent(group):
    threshold = group["Athlete average speed"].quantile(0.95)
    return group[group["Athlete average speed"] <= threshold]

df_filtered = df_fixed.groupby("Event name", group_keys=False).apply(remove_top_5_percent)

# Save the new dataset
df_filtered.to_csv("courses_2020_fixed_distance_km_no_top5.csv", index=False)


