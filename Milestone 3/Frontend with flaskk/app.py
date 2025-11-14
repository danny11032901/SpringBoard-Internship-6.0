from flask import Flask, render_template, request
import joblib
import csv
from datetime import datetime
import os
import traceback

app = Flask(__name__)

# ---------- Config ----------
MODEL_PATH = os.path.join("saved_model", "fake_job_model.pkl")
VECTORIZER_PATH = os.path.join("saved_model", "tfidf_vectorizer.pkl")
CSV_PATH = "predictions_log.csv"

# ---------- Load model & vectorizer ----------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"ERROR: Could not load model at '{MODEL_PATH}'.\n{e}")
    traceback.print_exc()
    raise SystemExit(f"Please put your trained model at: {MODEL_PATH}")

try:
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    print(f"ERROR: Could not load vectorizer at '{VECTORIZER_PATH}'.\n{e}")
    traceback.print_exc()
    raise SystemExit(f"Please put your vectorizer at: {VECTORIZER_PATH}")


# ---------- Save Predictions ----------
def append_prediction(job_description: str, label: str, confidence_percent: float):
    file_exists = os.path.isfile(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Job Description", "Prediction", "Confidence (%)"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            job_description,
            label,
            f"{confidence_percent:.2f}"
        ])


# ---------- ROUTES ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    job_description = request.form.get("job_description", "").strip()
    if not job_description:
        return render_template("index.html", error="Please enter a job description.")

    X = vectorizer.transform([job_description])

    pred = model.predict(X)[0]

    # confidence logic
    confidence_percent = 0.0
    try:
        proba = model.predict_proba(X)[0]
        confidence_percent = float(proba[pred]) * 100.0
    except:
        try:
            import numpy as np
            scores = model.decision_function(X)
            s = float(scores[0])
            conf = 1 / (1 + np.exp(-s))
            confidence_percent = conf * 100.0
        except:
            confidence_percent = 0.0

    label = "Fake Job" if int(pred) == 1 else "Real Job"

    append_prediction(job_description, label, confidence_percent)

    return render_template("result.html",
                           description=job_description,
                           label=label,
                           confidence=f"{confidence_percent:.2f}"
                           )


@app.route("/history")
def history():
    """Show ONLY LATEST 2 records — newest first."""
    if not os.path.exists(CSV_PATH):
        return render_template("history.html", header=None, rows=[], full=False)

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        data = list(csv.reader(f))

    if not data:
        return render_template("history.html", header=None, rows=[], full=False)

    header = data[0]
    rows = data[1:]

    # FIX ADDED ✔ — newest rows FIRST
    rows = rows[::-1]

    latest_two = rows[:2]

    return render_template("history.html", header=header, rows=latest_two, full=False)


@app.route("/history/full")
def history_full():
    """Show FULL history — newest first."""
    if not os.path.exists(CSV_PATH):
        return render_template("history.html", header=None, rows=[], full=True)

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        data = list(csv.reader(f))

    if not data:
        return render_template("history.html", header=None, rows=[], full=True)

    header = data[0]
    rows = data[1:]

    # FIX ADDED ✔ — newest first
    rows = rows[::-1]

    return render_template("history.html", header=header, rows=rows, full=True)


if __name__ == "__main__":
    app.run(debug=True)
