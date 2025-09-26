import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np

from src.complete_implementation import CreditCardFraudDetectionPipeline
from sklearn.metrics import roc_curve, confusion_matrix
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# In-memory user storage (for demo purposes)
users = {}  # {username: hashed_password}

# Load pipeline once
pipeline = CreditCardFraudDetectionPipeline()
pipeline.run_complete_pipeline(filepath=None, use_novel_sampling=False)
clf = pipeline.clf
scaler = pipeline.pre.scaler

# -------------------------
# Authentication Routes
# -------------------------


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        confirm = request.form['confirm_password'].strip()

        if username in users:
            flash('Username already exists!')
            return redirect(url_for('register'))
        if password != confirm:
            flash('Passwords do not match!')
            return redirect(url_for('register'))
        if not username or not password:
            flash('Please enter valid username and password.')
            return redirect(url_for('register'))

        # Hash password
        hashed = generate_password_hash(password)
        users[username] = hashed
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if username not in users or not check_password_hash(users[username], password):
            flash('Invalid username or password!')
            return redirect(url_for('login'))

        session['user'] = username
        flash(f'Welcome, {username}!')
        return redirect(url_for('home'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))


# -------------------------
# Main Prediction Routes
# -------------------------
@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['user'])


@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        features = [float(request.form[f'V{i}']) for i in range(1, 29)]
        features.append(float(request.form['Time']))
        features.append(float(request.form['Amount']))
    except Exception:
        flash('Please enter valid numeric values for all fields.')
        return redirect(url_for('home'))

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred, prob = clf.predict(X_scaled)
    label = 'Fraud' if pred[0] == 1 else 'Legitimate'
    prob = round(prob[0], 4)

    return render_template('result.html', label=label, probability=prob)


@app.route('/upload')
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html')


@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('upload'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('upload'))

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            required_cols = [f"V{i}" for i in range(
                1, 29)] + ['Time', 'Amount']
            for col in required_cols:
                if col not in df.columns:
                    flash(f"Missing required column: {col}")
                    return redirect(url_for('upload'))

            X = df[required_cols].values
            X_scaled = scaler.transform(X)
            preds, probs = clf.predict(X_scaled)
            df['Prediction'] = ['Fraud' if p ==
                                1 else 'Legitimate' for p in preds]
            df['Fraud_Probability'] = np.round(probs, 4)

            roc_data = None
            cm_data = None
            if 'Class' in df.columns:
                fpr, tpr, _ = roc_curve(df['Class'], probs)
                cm = confusion_matrix(df['Class'], preds)
                roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
                cm_data = cm.tolist()

            return render_template('batch_result.html',
                                   tables=[df.head(50).to_html(
                                       classes='data', index=False)],
                                   roc_data=roc_data,
                                   cm_data=cm_data,
                                   total=len(df))
        except Exception as e:
            flash(f"Error processing file: {str(e)}")
            return redirect(url_for('upload'))

    else:
        flash('Please upload a CSV file.')
        return redirect(url_for('upload'))


if __name__ == '__main__':
    app.run(debug=True)
