"""
=============================================================================
CAPSTONE PROJECT - GLOBAL STOCK MARKET ANALYTICS II
PHASE 6: FASTAPI DEPLOYMENT (v2: all five stocks, real CSV upload, Render-ready)
=============================================================================
Serves the finalized logistic regression model for each stock found in
model_files/ (MSFT, AAPL, GOOGL, NVDA, AMZN) as a REST API plus a web page.

ENDPOINTS
  GET  /                           web page (form + CSV upload)
  GET  /docs                       interactive API docs (FastAPI built-in)
  GET  /health                     service status and the tickers loaded
  GET  /tickers                    list of available tickers
  GET  /features?ticker=AMZN       the exact inputs a ticker's model expects
  POST /predict?ticker=AMZN        one prediction (JSON body of feature values)
  GET  /template/{ticker}.csv      a CSV template with the right columns
  POST /predict/batch              upload a CSV, get predictions back
  GET  /predict/batch/download/{f} download a batch result file
  POST /predict/batch/folder       predict every CSV in the server's data/ folder

MODEL FILES (put in model_files/, from submission_export.zip -> models/)
  <TICKER>_model.joblib    dict: {"model", "features", "mean", "std"}
  (the older single-model export amzn_model.joblib + amzn_model_metadata.json
   is also understood)

RUN LOCALLY
  pip install -r requirements.txt
  uvicorn main:app --reload
RUN ON RENDER
  start command: uvicorn main:app --host 0.0.0.0 --port $PORT
=============================================================================
"""

import glob
import io
import json
import os
import tempfile
import uuid
from datetime import datetime
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")
# Results go to the system temp dir: works on read-only/ephemeral hosts too.
PRED_DIR = os.path.join(tempfile.gettempdir(), "stock_predictions")
os.makedirs(PRED_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 5_000_000
MAX_ROWS = 50_000

app = FastAPI(
    title="Next-Open Direction Predictor",
    description="Predicts whether a stock opens above the previous close "
                "(Open(t) > Close(t-1)) for MSFT, AAPL, GOOGL, NVDA and AMZN. "
                "Educational project; not financial advice.",
    version="2.0.0",
)


# -----------------------------------------------------------------------
# Loading the models
# -----------------------------------------------------------------------

class Predictor:
    def __init__(self, ticker, model, features, mean=None, std=None):
        self.ticker = ticker
        self.model = model
        self.features = list(features)
        self.mean = pd.Series(mean)[self.features] if mean else None
        self.std = pd.Series(std)[self.features] if std else None

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features].astype(float)
        if self.mean is not None:
            X = (X - self.mean) / self.std
        return self.model.predict_proba(X)[:, 1]


PREDICTORS: Dict[str, Predictor] = {}


def load_models() -> None:
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*_model.joblib"))):
        prefix = os.path.basename(path).split("_model")[0]
        ticker = prefix.upper()
        try:
            obj = joblib.load(path)
            if isinstance(obj, dict) and "model" in obj:
                PREDICTORS[ticker] = Predictor(
                    ticker, obj["model"], obj["features"], obj.get("mean"), obj.get("std"))
            else:  # older export: bare estimator + separate metadata json
                meta_path = None
                for cand in (f"{prefix}_model_metadata.json", f"{prefix}_metadata.json"):
                    p = os.path.join(MODEL_DIR, cand)
                    if os.path.exists(p):
                        meta_path = p
                        break
                if meta_path is None:
                    raise FileNotFoundError(f"no metadata json for {os.path.basename(path)}")
                with open(meta_path) as f:
                    PREDICTORS[ticker] = Predictor(ticker, obj, json.load(f)["features"])
            print(f"Loaded {ticker}: {PREDICTORS[ticker].features}")
        except Exception as e:  # one bad file must not take the whole service down
            print(f"WARNING: could not load {path}: {e}")
    if not PREDICTORS:
        raise FileNotFoundError(
            f"No models found in {MODEL_DIR}. Copy the files from "
            "submission_export.zip -> models/ (the *_model.joblib files) into model_files/.")


load_models()


def get_predictor(ticker: Optional[str]) -> Predictor:
    if not ticker:
        return PREDICTORS[sorted(PREDICTORS)[0]]
    t = ticker.upper().strip()
    if t not in PREDICTORS:
        raise HTTPException(status_code=404,
                            detail=f"Unknown ticker '{ticker}'. Available: {sorted(PREDICTORS)}")
    return PREDICTORS[t]


def score_dataframe(df: pd.DataFrame, p: Predictor):
    """Adds predicted_probability / predicted_class. Rows with missing or
    non-numeric feature values are left blank rather than guessed."""
    X = df[p.features].apply(pd.to_numeric, errors="coerce")
    ok = X.notna().all(axis=1)
    probs = np.full(len(df), np.nan)
    if ok.any():
        probs[ok.values] = p.proba(X[ok])
    out = df.copy()
    out["predicted_probability"] = np.round(probs, 4)
    out["predicted_class"] = pd.Series(
        np.where(np.isnan(probs), np.nan, (probs >= 0.5).astype(float)), index=df.index
    ).astype("Int64")
    return out, probs, int(ok.sum())


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "tickers": sorted(PREDICTORS), "model_loaded": True}


@app.get("/tickers")
def tickers():
    return {"tickers": sorted(PREDICTORS)}


@app.get("/features")
def features(ticker: Optional[str] = Query(None)):
    p = get_predictor(ticker)
    return {"ticker": p.ticker, "features": p.features}


@app.post("/predict")
def predict(ticker: Optional[str] = Query(None),
            values: Dict[str, float] = Body(..., examples=[{"X15_Nikkei_PreOpen": 0.5}])):
    p = get_predictor(ticker)
    missing = [f for f in p.features if f not in values]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing feature values: {missing}")
    prob = float(p.proba(pd.DataFrame([{f: values[f] for f in p.features}]))[0])
    return {"ticker": p.ticker, "predicted_probability": round(prob, 4),
            "predicted_class": int(prob >= 0.5), "timestamp": datetime.now().isoformat()}


@app.get("/template/{ticker}.csv")
def template(ticker: str):
    p = get_predictor(ticker)
    csv = ",".join(["Date"] + p.features) + "\n" + ",".join(["2026-01-01"] + ["0"] * len(p.features)) + "\n"
    return Response(csv, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{p.ticker}_template.csv"'})


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...), ticker: Optional[str] = Form(None)):
    p = get_predictor(ticker)
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (limit 5 MB).")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read that file as CSV: {e}")
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="The CSV has no rows.")
    if len(df) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"Too many rows (limit {MAX_ROWS}).")
    missing = [f for f in p.features if f not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail={
            "message": f"Missing required columns for {p.ticker}", "missing": missing,
            "expected": p.features})

    out, probs, n_scored = score_dataframe(df, p)
    out_name = f"{p.ticker}_{uuid.uuid4().hex[:8]}_predictions.csv"
    out.to_csv(os.path.join(PRED_DIR, out_name), index=False)

    valid = probs[~np.isnan(probs)]
    preview = out.head(10).astype(object).where(out.head(10).notna(), None).to_dict(orient="records")
    return {
        "ticker": p.ticker,
        "source_file": os.path.basename(file.filename or "upload.csv"),
        "rows": len(df), "rows_scored": n_scored, "rows_skipped": len(df) - n_scored,
        "predicted_up": int((valid >= 0.5).sum()), "predicted_down": int((valid < 0.5).sum()),
        "mean_probability": round(float(valid.mean()), 4) if len(valid) else None,
        "download_url": f"/predict/batch/download/{out_name}",
        "preview": preview,
    }


@app.get("/predict/batch/download/{filename}")
def download(filename: str):
    safe = os.path.basename(filename)
    path = os.path.join(PRED_DIR, safe)
    if not safe.endswith("_predictions.csv") or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found (results are temporary).")
    return FileResponse(path, media_type="text/csv", filename=safe)


@app.post("/predict/batch/folder")
def predict_folder(ticker: Optional[str] = Query(None)):
    """Predicts every CSV in the server's data/ folder (the 'designated folder')."""
    p = get_predictor(ticker)
    files = [f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))]
    if not files:
        raise HTTPException(status_code=404, detail="No CSV files found in data/")
    summary = []
    for path in files:
        name = os.path.basename(path)
        try:
            df = pd.read_csv(path)
            missing = [f for f in p.features if f not in df.columns]
            if missing:
                summary.append({"file": name, "status": "SKIPPED", "reason": f"missing columns {missing}"})
                continue
            out, probs, n = score_dataframe(df, p)
            out_name = name.replace(".csv", "") + f"_{p.ticker}_predictions.csv"
            out.to_csv(os.path.join(PRED_DIR, out_name), index=False)
            summary.append({"file": name, "status": "PROCESSED", "rows_scored": n,
                            "output_file": out_name})
        except Exception as e:
            summary.append({"file": name, "status": "ERROR", "reason": str(e)})
    return {"ticker": p.ticker, "files": summary}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
