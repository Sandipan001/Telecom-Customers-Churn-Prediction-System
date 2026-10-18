
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "raw" / "data.csv"

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

print("\nMissing values:")
print(df.isnull().sum())

print("\nEmpty strings:")
print((df.astype("string").apply(lambda col: col.str.strip().eq("").sum())))

print("\nUnique customer IDs:")
if "customerID" in df.columns:
    print(df["customerID"].nunique())
    print("Duplicate customer IDs:", df["customerID"].duplicated().sum())
else:
    print("No customerID column found.")

print("\nNumeric conversion check:")
if "TotalCharges" in df.columns:
    converted = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print("Values that cannot be converted:", converted.isna().sum())
