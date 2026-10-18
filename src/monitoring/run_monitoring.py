
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
prediction_log = project_root / "logs" / "predictions.csv"
monitor_script = project_root / "src" / "monitoring" / "monitor_drift.py"

if not prediction_log.exists():
    print("No prediction log found. Monitoring skipped.")
    raise SystemExit(0)

import pandas as pd

records = pd.read_csv(prediction_log)

if len(records) < 100:
    print(f"Monitoring deferred: {len(records)} records available; 100 required.")
    raise SystemExit(0)

result = subprocess.run(
    [sys.executable, str(monitor_script)],
    cwd=project_root,
    check=False
)

if result.returncode != 0:
    print("Monitoring failed. Retraining was not started.")
    raise SystemExit(result.returncode)

print("Monitoring completed.")
print("Review drift results before deciding whether retraining is appropriate.")
print("No model was replaced by this monitoring run.")
