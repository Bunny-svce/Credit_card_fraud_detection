💳 Credit Card Fraud Detection Web App

An interactive Flask-based web application leveraging a multi-stage machine learning framework to detect fraudulent credit card transactions.
The app supports both single transaction predictions and batch CSV uploads, with dynamic charts for interpretability.

🚀 Features

🔍 Multi-model ensemble fraud detection pipeline.

🧾 Single transaction prediction with validation & probability score.

📂 Batch CSV upload for bulk detection (up to thousands of transactions).

📊 Interactive ROC Curve & Confusion Matrix (heatmap + metrics).

🎨 Responsive UI with Bootstrap, FontAwesome icons, and animations.

🐳 Dockerized for portable, production-ready deployment.

⚡ Gunicorn + NGINX configuration included for scalability.

🛠️ Tech Stack

Backend: Python, Flask

Frontend: HTML5, CSS3, Bootstrap, Chart.js

Machine Learning: Scikit-learn, Pandas, NumPy

Deployment: Docker, Gunicorn, NGINX

📂 Project Structure
credit_card_fraud_detection/
│
├── app.py                  # Flask web application
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Multi-container deployment (optional)
├── nginx.conf              # NGINX reverse proxy (optional)
├── requirements.txt        # Python dependencies
├── sample_input.csv        # CSV template for batch upload
│
├── src/                    # Core ML pipeline
│   └── complete_implementation.py
│
├── templates/              # HTML templates
│   ├── index.html
│   ├── result.html
│   ├── upload.html
│   ├── batch_result.html
│
├── static/
│   ├── css/
│   │   └── style.css       # Custom styles
│   └── images/
│       └── logo.png        # Project logo
│
└── screenshots/            # Screenshots for README
    ├── home.png
    └── batch_results.png

⚙️ Setup Instructions
🔹 Local Development
1. Clone the repository
git clone <repo_url>
cd credit_card_fraud_detection

2. Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements
