
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "processed" / "cleaned_data.csv"

df = pd.read_csv(data_path)

df["ChargeGroup"] = pd.qcut(
    df["MonthlyCharges"],
    q=4,
    duplicates="drop"
)

charge_analysis = (
    df.groupby("ChargeGroup", observed=True)
    .agg(
        Total_Customers=("Churn", "size"),
        Churned_Customers=("Churn", lambda x: (x == "Yes").sum()),
        Churn_Rate=("Churn", lambda x: (x == "Yes").mean() * 100),
        Average_Monthly_Charges=("MonthlyCharges", "mean")
    )
)

charge_analysis["Churn_Rate"] = charge_analysis["Churn_Rate"].round(2)
charge_analysis["Average_Monthly_Charges"] = (
    charge_analysis["Average_Monthly_Charges"].round(2)
)

print("Churn analysis by monthly charges:")
print(charge_analysis)
