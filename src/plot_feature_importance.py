import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

imp = pd.read_csv("reports/feature_importance.csv", header=None)
if imp.shape[1] == 1:
    imp = imp.reset_index()
imp.columns = ["feature", "importance"]

imp = imp.dropna(subset=["feature", "importance"])
imp["feature"] = imp["feature"].astype(str)
imp["importance"] = pd.to_numeric(imp["importance"], errors="coerce")
imp = imp.dropna(subset=["importance"])
imp = imp.sort_values("importance", ascending=True).tail(15)

plt.figure(figsize=(8, 6))
plt.barh(imp["feature"], imp["importance"])
plt.title("Feature Importance (LightGBM)")
plt.xlabel("Importance")
plt.tight_layout()

Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/feature_importance.png", dpi=150)
print("saved → reports/figures/feature_importance.png")
