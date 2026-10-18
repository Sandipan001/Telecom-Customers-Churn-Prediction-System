
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "processed" / "cleaned_data.csv"

df = pd.read_csv(data_path)

churn_counts = df["Churn"].value_counts()
churn_percentages = df["Churn"].value_counts(normalize=True) * 100

print("Total customers:", len(df))

print("\nCustomer churn counts:")
print(churn_counts)

print("\nCustomer churn percentages:")
print(churn_percentages.round(2))

print("\nCustomers who left:", churn_counts.get("Yes", 0))
print("Customers who stayed:", churn_counts.get("No", 0))

print(
    "\nOverall churn rate:",
    round((df["Churn"] == "Yes").mean() * 100, 2),
    "%"
)
