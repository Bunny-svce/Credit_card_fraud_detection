import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

from src.complete_implementation import CreditCardFraudDetectionPipeline
from sklearn.metrics import roc_curve, confusion_matrix

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load pipeline once
pipeline = CreditCardFraudDetectionPipeline()
pipeline.run_complete_pipeline(filepath=None, use_novel_sampling=False)
clf = pipeline.clf
scaler = pipeline.pre.scaler


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
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
    return render_template('upload.html')


@app.route('/upload_predict', methods=['POST'])
def upload_predict():
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
