
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[2]
data_path = project_root / "data" / "processed" / "cleaned_data.csv"

df = pd.read_csv(data_path)

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 36, float("inf")],
    labels=["New (0-12 months)", "Medium (13-36 months)", "Long-term (37+ months)"]
)

tenure_analysis = (
    df.groupby("TenureGroup", observed=True)
    .agg(
        Total_Customers=("Churn", "size"),
        Churned_Customers=("Churn", lambda x: (x == "Yes").sum()),
        Churn_Rate=("Churn", lambda x: (x == "Yes").mean() * 100)
    )
)

tenure_analysis["Churn_Rate"] = tenure_analysis["Churn_Rate"].round(2)

print("Churn analysis by customer tenure:")
print(tenure_analysis)
