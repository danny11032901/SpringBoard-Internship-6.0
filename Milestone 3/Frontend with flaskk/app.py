from flask import Flask, render_template, request, redirect, url_for
import joblib
import csv
from datetime import datetime
import os

app = Flask(__name__)

# ------------------------------------------------
# Load REAL Machine Learning Model + TF-IDF
# ------------------------------------------------
model = joblib.load("saved_model/fake_job_model.pkl")
vectorizer = joblib.load("saved_model/tfidf_vectorizer.pkl")

CSV_FILE = "predictions_log.csv"


# ------------------------------------------------
# Save prediction to CSV
# ------------------------------------------------
def save_to_csv(job_description, prediction, confidence):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header only once
        if not file_exists:
            writer.writerow(["Timestamp", "Job Description", "Prediction", "Confidence"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            job_description,
            prediction,
            f"{confidence:.2f}%"
        ])


# ------------------------------------------------
# HOME PAGE
# ------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------
# PREDICT PAGE
# ------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    job_description = request.form["job_description"]

    # Transform text → TF-IDF
    X = vectorizer.transform([job_description])

    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = max(proba) * 100

    label = "Fake Job" if pred == 1 else "Real Job"

    # Save to history CSV
    save_to_csv(job_description, label, confidence)

    return render_template(
        "result.html",
        description=job_description,
        label=label,
        confidence=f"{confidence:.2f}"
    )


# ------------------------------------------------
# HISTORY (ONLY LAST 2)
# ------------------------------------------------
@app.route("/history")
def history():
    data = []

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            data = list(csv.reader(f))

    # Header + last 2 entries
    latest_two = [data[0]] + data[-2:] if len(data) > 2 else data

    return render_template("history.html", data=latest_two, full=False)


# ------------------------------------------------
# FULL HISTORY
# ------------------------------------------------
@app.route("/history/full")
def full_history():
    data = []

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            data = list(csv.reader(f))

    return render_template("history.html", data=data, full=True)


# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
