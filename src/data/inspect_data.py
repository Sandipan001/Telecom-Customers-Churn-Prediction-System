from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "raw" / "data.csv"

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

if "Churn" in df.columns:
    print("\nChurn distribution:")
    print(df["Churn"].value_counts(dropna=False))
else:
    print("\nNo column named 'Churn' found.")
