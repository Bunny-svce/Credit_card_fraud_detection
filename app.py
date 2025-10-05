import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
from werkzeug.security import generate_password_hash, check_password_hash

# Import ML pipeline
from src.complete_implementation import CreditCardFraudDetectionPipeline

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# -------------------------
# In-memory user storage (temporary for demo)
# -------------------------
users = {}  # {username: hashed_password}

# -------------------------
# Load model pipeline once
# -------------------------
pipeline = CreditCardFraudDetectionPipeline()
pipeline.run_complete_pipeline(filepath=None, use_novel_sampling=False)
clf = pipeline.clf
scaler = pipeline.pre.scaler


# -------------------------
# Landing Page (Before Login)
# -------------------------
@app.route('/')
def welcome():
    """Landing page shown before login."""
    return render_template('home.html')


# -------------------------
# Authentication Routes
# -------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register new user."""
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        confirm = request.form['confirm_password'].strip()

        # Validation checks
        if username in users:
            flash('Username already exists!')
            return redirect(url_for('register'))
        if password != confirm:
            flash('Passwords do not match!')
            return redirect(url_for('register'))
        if not username or not password:
            flash('Please enter valid username and password.')
            return redirect(url_for('register'))

        # Hash and store password
        hashed = generate_password_hash(password)
        users[username] = hashed
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if username not in users or not check_password_hash(users[username], password):
            flash('Invalid username or password!')
            return redirect(url_for('login'))

        session['user'] = username
        flash(f'Welcome, {username}!')
        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Log out user."""
    session.pop('user', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))


# -------------------------
# Dashboard (Main Prediction Page)
# -------------------------
@app.route('/dashboard')
def dashboard():
    """Main fraud prediction dashboard."""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['user'])


# -------------------------
# Single Transaction Prediction
# -------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Predict fraud for a single transaction."""
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        # Collect input features
        features = [float(request.form[f'V{i}']) for i in range(1, 29)]
        features.append(float(request.form['Time']))
        features.append(float(request.form['Amount']))
    except Exception:
        flash('Please enter valid numeric values for all fields.')
        return redirect(url_for('dashboard'))

    # Scale and predict
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    preds, probs = clf.predict(X_scaled)
    label = 'Fraud' if preds[0] == 1 else 'Legitimate'
    prob = round(probs[0], 4)

    return render_template('result.html', label=label, probability=prob)


# -------------------------
# Batch Upload & Prediction
# -------------------------
@app.route('/upload')
def upload():
    """Upload CSV for batch fraud detection."""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('upload.html')


@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    """Process batch CSV and show results."""
    if 'user' not in session:
        return redirect(url_for('login'))

    # Validate upload
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('upload'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('upload'))

    # Process CSV
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            required_cols = [f"V{i}" for i in range(
                1, 29)] + ['Time', 'Amount']

            # Validate column presence
            for col in required_cols:
                if col not in df.columns:
                    flash(f"Missing required column: {col}")
                    return redirect(url_for('upload'))

            # Predict
            X = df[required_cols].values
            X_scaled = scaler.transform(X)
            preds, probs = clf.predict(X_scaled)
            df['Prediction'] = ['Fraud' if p ==
                                1 else 'Legitimate' for p in preds]
            df['Fraud_Probability'] = np.round(probs, 4)

            # Optional: Compute ROC and Confusion Matrix if labels present
            roc_data, cm_data = None, None
            if 'Class' in df.columns:
                fpr, tpr, _ = roc_curve(df['Class'], probs)
                cm = confusion_matrix(df['Class'], preds)
                roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
                cm_data = cm.tolist()

            return render_template(
                'batch_result.html',
                tables=[df.head(10000).to_html(classes='data', index=False)],
                roc_data=roc_data,
                cm_data=cm_data,
                total=len(df)
            )

        except Exception as e:
            flash(f"Error processing file: {str(e)}")
            return redirect(url_for('upload'))
    else:
        flash('Please upload a valid CSV file.')
        return redirect(url_for('upload'))


# -------------------------
# Run the App
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
