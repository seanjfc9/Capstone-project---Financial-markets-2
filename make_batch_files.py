"""
Builds ready-to-upload batch CSV files for the web app, one per stock.

Run from the phase6_deployment folder AFTER you have:
  - model_files/<TICKER>_model.joblib   (from submission_export.zip -> models/)
  - modeling_dataset.csv                (from submission_export.zip), placed
                                         next to this script

For each stock it writes data/<TICKER>_test_rows.csv with the last 20% of
that stock's modelling rows (the test period): Date, the model's feature
columns, and the ACTUAL outcome (Target_Open_Direction). Upload one to the
web page's batch section and compare predicted_class with the actual column.

    python make_batch_files.py
"""
import glob
import os

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(HERE, "modeling_dataset.csv")
if not os.path.exists(src):
    raise SystemExit("Put modeling_dataset.csv (from submission_export.zip) next to this script.")

df = pd.read_csv(src, parse_dates=["Date"])
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)

for path in sorted(glob.glob(os.path.join(HERE, "model_files", "*_model.joblib"))):
    obj = joblib.load(path)
    if not (isinstance(obj, dict) and "features" in obj):
        print(f"skip {os.path.basename(path)} (old format)")
        continue
    t = os.path.basename(path).split("_model")[0].upper()
    sub = df[df["Ticker"] == t].sort_values("Date")
    test = sub.iloc[int(len(sub) * 0.8):]
    cols = ["Date"] + obj["features"] + ["Target_Open_Direction"]
    out = os.path.join(HERE, "data", f"{t}_test_rows.csv")
    test[cols].to_csv(out, index=False)
    print(f"{t}: wrote {len(test)} rows -> data/{t}_test_rows.csv")
