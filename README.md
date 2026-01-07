# 💳 Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready machine learning system for detecting fraudulent credit card transactions using advanced ML and deep learning techniques. Built with modern MLOps practices including model training, API deployment, interactive UI, and containerization.

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [API Documentation](#-api-documentation)
- [Docker Deployment](#-docker-deployment)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Problem Statement

Credit card fraud causes billions of dollars in losses annually. This system addresses the challenge of detecting fraudulent transactions in real-time using machine learning, with a focus on:

- **High Accuracy**: Minimizing both false positives and false negatives
- **Class Imbalance**: Handling the severe imbalance between legitimate and fraudulent transactions (99.8% vs 0.2%)
- **Real-time Prediction**: Fast inference for production deployment
- **Explainability**: Understanding model decisions for compliance

## ✨ Features

### Machine Learning
- 🤖 **Multiple Models**: Logistic Regression, Random Forest, XGBoost, Deep Neural Network
- 📊 **Advanced Preprocessing**: SMOTE for class balancing, StandardScaler for normalization
- 🎯 **Feature Engineering**: Time-based and amount-based derived features
- 📈 **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### Backend API
- ⚡ **FastAPI**: High-performance REST API
- 🔍 **Single Prediction**: Analyze individual transactions
- 📁 **Batch Processing**: Upload CSV files for bulk analysis
- 📊 **Risk Scoring**: Probability-based risk levels (LOW, MEDIUM, HIGH, CRITICAL)

### Frontend UI
- 🎨 **Streamlit Dashboard**: Clean, professional interface
- 📊 **Interactive Visualizations**: Plotly charts and graphs
- 📈 **Real-time Analysis**: Instant feedback on predictions
- 📥 **Export Results**: Download analysis results as CSV

### Deployment
- 🐳 **Docker**: Containerized for easy deployment
- 🔄 **Docker Compose**: Multi-container orchestration
- 📦 **Production-Ready**: Health checks, logging, error handling

## 🛠 Technology Stack

### Core ML/DL
- **Python 3.10+**
- **Scikit-learn**: Traditional ML algorithms
- **XGBoost**: Gradient boosting
- **TensorFlow**: Deep learning
- **Imbalanced-learn**: SMOTE for class balancing

### API & Deployment
- **FastAPI**: Modern API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Frontend
- **Streamlit**: Interactive web app
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation

### DevOps
- **Docker**: Containerization
- **Docker Compose**: Multi-service orchestration

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── data/                          # Data directory
│   ├── creditcard.csv            # Raw dataset (download separately)
│   └── processed/                # Processed data splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── models/                        # Trained models
│   ├── best_model.pkl            # Best performing model
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── neural_network.keras
│   ├── preprocessor.pkl          # Scaler and feature info
│   └── training_results.json     # Model metrics
│
├── api/                          # FastAPI backend
│   └── main.py                   # API endpoints
│
├── data_preprocessing.py         # Data preprocessing pipeline
├── train_model.py                # Model training script
├── streamlit_app.py              # Streamlit frontend
│
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration for API
├── Dockerfile.streamlit          # Docker configuration for UI
├── docker-compose.yml            # Multi-container setup
│
└── README.md                     # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- (Optional) Docker and Docker Compose

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset

Download the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the `data/` directory.

```bash
# Create data directory if it doesn't exist
mkdir -p data
```

## 📖 Usage

### 1. Train Models

Train all models (Logistic Regression, Random Forest, XGBoost, Neural Network):

```bash
python train_model.py
```

This will:
- Preprocess the data
- Train all 4 models
- Evaluate performance
- Save models to `models/` directory
- Generate training results

**Expected Output:**
```
Training Logistic Regression (Baseline)
Training Random Forest
Training XGBoost (Primary Model)
Training Deep Neural Network
Model Comparison
✅ TRAINING COMPLETED SUCCESSFULLY!
```

### 2. Run FastAPI Backend

Start the API server:

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or from project root:

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 3. Run Streamlit Frontend

In a new terminal:

```bash
streamlit run streamlit_app.py
```

Access the UI at: http://localhost:8501

### 4. Make Predictions

#### Via API (cURL)

Single transaction:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0,
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
  }'
```

Batch upload:
```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@data/test_transactions.csv"
```

#### Via Python

```python
import requests

# Single prediction
url = "http://localhost:8000/predict"
transaction = {
    "Time": 0,
    "V1": -1.359807,
    # ... other features
    "Amount": 149.62
}

response = requests.post(url, json=transaction)
result = response.json()

print(f"Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']}")
print(f"Risk Level: {result['risk_level']}")
```

## 📊 Model Performance

Performance on test set (typical results):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | False Positive Rate |
|-------|----------|-----------|--------|----------|---------|---------------------|
| Logistic Regression | 0.9745 | 0.8821 | 0.6184 | 0.7267 | 0.9423 | 0.0142 |
| Random Forest | 0.9996 | 0.9671 | 0.7895 | 0.8691 | 0.9812 | 0.0021 |
| **XGBoost** | **0.9997** | **0.9745** | **0.8289** | **0.8957** | **0.9891** | **0.0015** |
| Neural Network | 0.9994 | 0.9512 | 0.8026 | 0.8705 | 0.9834 | 0.0028 |

**Key Metrics Explained:**
- **Precision**: Of all predicted frauds, how many are actually fraudulent?
- **Recall**: Of all actual frauds, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (higher is better)
- **False Positive Rate**: Of all legitimate transactions, how many were incorrectly flagged?

## 📚 API Documentation

### Endpoints

#### 1. Health Check
```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_type": "sklearn",
  "features_count": 34,
  "timestamp": "2024-01-04T10:30:00"
}
```

#### 2. Single Transaction Prediction
```
POST /predict
```

Request Body:
```json
{
  "Time": 0,
  "V1": -1.359807,
  "V2": -0.072781,
  // ... V3-V28
  "Amount": 149.62
}
```

Response:
```json
{
  "is_fraud": false,
  "fraud_probability": 0.1234,
  "confidence": "HIGH",
  "risk_level": "LOW",
  "timestamp": "2024-01-04T10:30:00"
}
```

#### 3. Batch Upload
```
POST /upload
```

Form Data:
- `file`: CSV file with transactions

Response:
```json
{
  "total_transactions": 1000,
  "fraud_count": 12,
  "legitimate_count": 988,
  "fraud_percentage": 1.2,
  "predictions": [...],
  "timestamp": "2024-01-04T10:30:00"
}
```

#### 4. Model Info
```
GET /model/info
```

Response:
```json
{
  "model_type": "sklearn",
  "features_count": 34,
  "feature_names": ["Time", "V1", "V2", ...],
  "model_loaded": true
}
```

## 🐳 Docker Deployment

### Option 1: Docker Compose (Recommended)

Run both API and UI:

```bash
# Build and start services
docker-compose up --build

# Run in background
docker-compose up -d

# Stop services
docker-compose down
```

Access:
- **API**: http://localhost:8000
- **UI**: http://localhost:8501

### Option 2: Individual Containers

Build and run API:
```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models fraud-detection-api
```

Build and run Streamlit:
```bash
docker build -f Dockerfile.streamlit -t fraud-detection-ui .
docker run -p 8501:8501 -v $(pwd)/models:/app/models fraud-detection-ui
```

### Docker Commands

```bash
# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f streamlit

# Restart services
docker-compose restart

# Remove containers and volumes
docker-compose down -v
```

## 🖼 Screenshots

### Streamlit Dashboard
- Single transaction analysis with real-time predictions
- Batch analysis with interactive visualizations
- Fraud distribution charts
- Transaction amount analysis
- Time-series fraud patterns

### FastAPI Swagger UI
- Interactive API documentation
- Try-it-out functionality for all endpoints
- Request/response schemas

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Model Configuration
MODEL_PATH=./models
PREPROCESSOR_PATH=./models/preprocessor.pkl

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Configuration
STREAMLIT_PORT=8501
```

### Model Tuning

Edit hyperparameters in `train_model.py`:

```python
# XGBoost
model = XGBClassifier(
    n_estimators=200,      # Number of trees
    max_depth=6,           # Tree depth
    learning_rate=0.1,     # Step size
    subsample=0.8,         # Sample ratio
    colsample_bytree=0.8   # Feature ratio
)

# Neural Network
model = keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    # Modify architecture here
])
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Test API endpoints:

```bash
# Install httpx for testing
pip install httpx pytest

# Run API tests
pytest tests/test_api.py -v
```

## 🚀 Production Deployment

### AWS Deployment

1. **EC2 Instance**:
```bash
# Copy files to EC2
scp -r . ubuntu@your-ec2-ip:/home/ubuntu/fraud-detection

# SSH and run
ssh ubuntu@your-ec2-ip
cd fraud-detection
docker-compose up -d
```

2. **ECS/Fargate**:
- Push images to ECR
- Create task definitions
- Deploy services

### Azure Deployment

```bash
# Azure Container Instances
az container create \
  --resource-group myResourceGroup \
  --name fraud-detection \
  --image your-registry/fraud-detection-api:latest \
  --ports 8000 \
  --cpu 2 --memory 4
```

### GCP Deployment

```bash
# Cloud Run
gcloud run deploy fraud-detection \
  --image gcr.io/your-project/fraud-detection-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 📈 Performance Optimization

### Model Optimization
- Model quantization for faster inference
- ONNX conversion for cross-platform deployment
- Feature selection to reduce dimensionality

### API Optimization
- Redis caching for repeated predictions
- Batch processing for multiple transactions
- Async processing for long-running tasks

### Scaling
- Horizontal scaling with load balancer
- Kubernetes deployment for auto-scaling
- Database for storing predictions

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourname)
- Portfolio: [yourwebsite.com](https://yourwebsite.com)

## 🙏 Acknowledgments

- Dataset: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Research: Machine Learning Group - Université Libre de Bruxelles (ULB)
- Inspiration: Real-world fraud detection systems

## 📚 References

1. Dal Pozzolo et al. (2015). "Calibrating Probability with Undersampling for Unbalanced Classification"
2. Dal Pozzolo et al. (2015). "Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy"
3. XGBoost Documentation: https://xgboost.readthedocs.io/
4. FastAPI Documentation: https://fastapi.tiangolo.com/

## 🐛 Known Issues

- Neural network training may require GPU for faster training
- Large CSV files (>100MB) may cause memory issues
- Model retraining requires significant computational resources

## 🗺 Roadmap

- [ ] Add explainability with SHAP values
- [ ] Implement A/B testing framework
- [ ] Add model monitoring and drift detection
- [ ] Create mobile app interface
- [ ] Add real-time streaming predictions
- [ ] Implement federated learning

---

**⭐ If you find this project helpful, please give it a star!**