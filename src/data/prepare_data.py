
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "processed" / "cleaned_data.csv"

df = pd.read_csv(data_path)

target_column = "Churn"
id_column = "customerID"

print("Dataset shape:", df.shape)

print("\nTarget values:")
print(df[target_column].value_counts(dropna=False))

print("\nTarget data type:")
print(df[target_column].dtype)

print("\nUnique target values:")
print(df[target_column].unique())

print("\nCustomer ID examples:")
print(df[id_column].head())

print("\nDuplicate customer IDs:", df[id_column].duplicated().sum())

print("\nFeature data types:")
print(df.drop(columns=[target_column, id_column]).dtypes)

print("\nTarget missing values:", df[target_column].isna().sum())
