from flask import Flask, render_template, request, redirect, session
import joblib
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = "mysecretkey123"   # required for admin login session


# -------------------------------------------------
# LOAD MODEL + VECTORIZER
# -------------------------------------------------
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


# -------------------------------------------------
# INITIALIZE DATABASE (predictions + admin)
# -------------------------------------------------
def init_db():
    conn = sqlite3.connect('job_predictions.db')

    # Table for predictions
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_description TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    # Table for admin login
    conn.execute('''
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        );
    ''')

    # Insert default admin if not exists
    cursor = conn.execute("SELECT * FROM admin WHERE username='admin'")
    if not cursor.fetchone():
        conn.execute("INSERT INTO admin (username, password) VALUES ('admin', 'admin123')")
        print("Default admin created: admin / admin123")

    conn.commit()
    conn.close()


init_db()



# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')



# -------------------------------------------------
# PREDICT ROUTE
# -------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    job_desc = request.form['job_description'].strip()

    if not job_desc or len(job_desc.split()) < 5:
        return render_template('index.html',
                               error="⚠ Please enter a detailed job description (min 5 words).")

    # Predict
    X = vectorizer.transform([job_desc])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    label = "Fake Job" if pred == 1 else "Real Job"
    confidence = round(prob * 100, 2) if pred == 1 else round((1 - prob) * 100, 2)

    # Save into DB
    conn = sqlite3.connect('job_predictions.db')
    conn.execute(
        'INSERT INTO predictions (job_description, prediction, confidence) VALUES (?, ?, ?)',
        (job_desc, label, confidence)
    )
    conn.commit()
    conn.close()

    return render_template('result.html',
                           label=label,
                           confidence=confidence,
                           description=job_desc)



# -------------------------------------------------
# HISTORY PAGE
# -------------------------------------------------
@app.route('/history')
def history():
    conn = sqlite3.connect('job_predictions.db')
    cursor = conn.execute(
        'SELECT job_description, prediction, confidence, timestamp FROM predictions ORDER BY id DESC'
    )
    records = cursor.fetchall()
    conn.close()

    return render_template('history.html', records=records)



# -------------------------------------------------
# ADMIN LOGIN
# -------------------------------------------------
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():

    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('job_predictions.db')
        cursor = conn.execute(
            "SELECT * FROM admin WHERE username=? AND password=?",
            (username, password)
        )
        admin = cursor.fetchone()
        conn.close()

        if admin:
            session['admin_logged_in'] = True
            return redirect('/admin_dashboard')
        else:
            return render_template('admin_login.html', error="Invalid username or password")

    return render_template('admin_login.html')



# -------------------------------------------------
# ADMIN DASHBOARD (PROTECTED)
# -------------------------------------------------
@app.route('/admin_dashboard')
def admin_dashboard():

    if not session.get('admin_logged_in'):
        return redirect('/admin_login')

    conn = sqlite3.connect('job_predictions.db')
    fake_jobs = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'"
    ).fetchone()[0]

    real_jobs = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'"
    ).fetchone()[0]

    total = fake_jobs + real_jobs
    conn.close()

    return render_template('admin_dashboard.html',
                           total=total,
                           fake=fake_jobs,
                           real=real_jobs)



# -------------------------------------------------
# ADMIN LOGOUT
# -------------------------------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/admin_login')



# -------------------------------------------------
# RUN APP
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
