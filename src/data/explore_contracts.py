
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "processed" / "cleaned_data.csv"

df = pd.read_csv(data_path)

contract_analysis = (
    df.groupby("Contract", observed=True)
    .agg(
        Total_Customers=("Churn", "size"),
        Churned_Customers=("Churn", lambda x: (x == "Yes").sum()),
        Churn_Rate=("Churn", lambda x: (x == "Yes").mean() * 100)
    )
    .sort_values("Churn_Rate", ascending=False)
)

contract_analysis["Churn_Rate"] = contract_analysis["Churn_Rate"].round(2)

print("Churn analysis by contract type:")
print(contract_analysis)
