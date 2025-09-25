# Credit Card Fraud Detection Web App

![Project Logo](screenshots/logo.png)

An interactive web application leveraging a multi-stage machine learning framework to detect credit card fraud transactions. The app supports both single transaction predictions and batch CSV uploads, displaying results with interactive visuals.

---

## Features

- Multi-model ensemble fraud detection pipeline.
- Single transaction input with validation & immediate prediction.
- Batch CSV upload for bulk fraud detection with summary stats.
- Dynamic ROC curve and confusion matrix visualizations.
- Responsive UI with Bootstrap, FontAwesome icons, and animations.
- Dockerized for easy deployment and scalability.
- Ready for production with Gunicorn and NGINX reverse proxy.

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)

### Local Development Setup

1. Clone the repository:
   git clone <repo_url>
   cd credit_card_fraud_detection

2. Create and activate virtual environment:
   python3 -m venv venv
   source venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt


4. Run the Flask application:
   python app.py


5. Open your browser to [http://localhost:5000](http://localhost:5000).

---

### Using Docker

1. Build the Docker image:
   docker build -t credit_fraud_app .


2. Run the Docker container:
   docker run -p 8000:8000 credit_fraud_app


3. Access the app at [http://localhost:8000](http://localhost:8000).

---

## Usage Guide

### Single Transaction Prediction

- Input the 28 anonymized features (`V1` to `V28`), along with transaction `Time` and `Amount`.
- Submit to receive a fraud or legitimate prediction with probability score.

### Batch Prediction

- Navigate to the Upload CSV page.
- Upload a CSV file containing columns `V1` to `V28`, `Time`, `Amount`.
- Preview predictions in a table along with ROC curve and confusion matrix charts.

---

## Sample CSV

Download and use `sample_input.csv` (provided in repo) as a template for batch upload. It includes headers and example values for all required fields.

---

## Screenshots

![Home page](screenshots/home.png)
![Batch results](screenshots/batch_results.png)

---

## Folder Structure

credit_card_fraud_detection/
│
├── app.py # Flask web application
├── Dockerfile # Docker configuration
├── docker-compose.yml # Compose for multi-container deployment (optional)
├── nginx.conf # Nginx reverse proxy (optional)
├── requirements.txt # Python dependencies
├── sample_input.csv # CSV template for batch upload
│
├── src/ # Core ML pipeline code
│ └── complete_implementation.py
│
├── templates/ # HTML templates
│ ├── index.html
│ ├── result.html
│ ├── upload.html
│ ├── batch_result.html
│
├── static/
│ ├── css/
│ │ └── style.css # Custom styles
│ └── images/
│ └── logo.png # Project logo or favicon
│
└── screenshots/ # Screenshot images for this README


---

## Contributing

Contributions warmly welcomed! Please submit pull requests or file issues for improvements.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or support, please open an issue.

---




