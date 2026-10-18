
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
raw_path = project_root / "data" / "raw" / "data.csv"
processed_path = project_root / "data" / "processed" / "cleaned_data.csv"

df = pd.read_csv(raw_path)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype("string").str.strip(),
    errors="coerce"
)
df["TotalCharges"] = df["TotalCharges"].fillna(0)


print("Missing values remaining:", df.isnull().sum().sum())
print("Missing TotalCharges:", df["TotalCharges"].isna().sum())
print("Zero-tenure customers:", (df["tenure"] == 0).sum())
print("Dataset shape:", df.shape)

print("Missing TotalCharges after conversion:", df["TotalCharges"].isna().sum())

print("\nAffected customer records:")
print(df.loc[df["TotalCharges"].isna(), ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]])

print("\nTenure values for affected records:")
print(df.loc[df["TotalCharges"].isna(), "tenure"].value_counts(dropna=False))

processed_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(processed_path, index=False)

print("\nCleaned dataset saved to:", processed_path)
print("Dataset shape:", df.shape)
print("TotalCharges data type:", df["TotalCharges"].dtype)
