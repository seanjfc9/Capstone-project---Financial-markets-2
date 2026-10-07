# Next-Open Direction Predictor (Phase 6)

A FastAPI app that serves the final logistic regression model for MSFT, AAPL,
GOOGL, NVDA and AMZN: a web page for single predictions, CSV upload for batch
predictions, and interactive API docs at `/docs`.

```
phase6_deployment/
├── main.py              the FastAPI app
├── requirements.txt
├── render.yaml          Render deployment settings
├── make_batch_files.py  builds ready-to-upload test CSVs from modeling_dataset.csv
├── model_files/         put the five <TICKER>_model.joblib files here
├── data/                sample CSVs (amzn_sample_week.csv is included)
└── static/index.html    the web page
```

## 1. Add the models (required)

From `submission_export.zip`, copy everything in its `models/` folder that ends
in `_model.joblib` into `model_files/`. The app will not start without them.
(The `*_metadata.json` files are documentation; the app does not need them.)

## 2. Run it on your computer

```
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/ for the page and http://127.0.0.1:8000/docs for the API docs.

## 3. Make batch files to upload (optional)

Copy `modeling_dataset.csv` (from `submission_export.zip`) next to `main.py`, then:

```
python make_batch_files.py
```

This writes `data/<TICKER>_test_rows.csv` for each stock: the test-period
rows with the model's input columns and the actual outcome
(`Target_Open_Direction`). Upload one on the web page and compare
`predicted_class` against the actual column. `data/amzn_sample_week.csv` is a
5-row example for AMZN. The web page also has a "Download a CSV template" link
for any stock.

## 4. Put it on Render (public link)

1. Create a GitHub repository and upload this whole folder, **including the
   five `.joblib` files in `model_files/`** (they are small).
2. On render.com: New > Web Service > connect that repository.
   - If it asks: Runtime **Python**, Build command `pip install -r requirements.txt`,
     Start command `uvicorn main:app --host 0.0.0.0 --port $PORT`.
     (`render.yaml` already holds these, plus Python 3.11.9.)
3. Deploy. When it is live, open `https://<your-service>.onrender.com/` and `/docs`.

Notes
- The free plan sleeps after inactivity; the first request after a break can take about a minute.
- Uploaded results are temporary files and disappear when the service restarts.
- If the logs say the model could not be loaded, the scikit-learn version differs
  from the one that saved it. In Colab run `import sklearn; print(sklearn.__version__)`,
  then change the line in `requirements.txt` to `scikit-learn==<that version>`.

## What the models expect

Each stock's model uses a few columns (see `/features?ticker=AMZN`, or the
template download). The Nikkei and futures columns are the same-day returns in
percent (for example `0.48` means +0.48%). Note that in training, the futures
column is the daily close-to-close return of the E-mini contract, so for the
most faithful use enter the full-day value rather than a pre-open reading.
