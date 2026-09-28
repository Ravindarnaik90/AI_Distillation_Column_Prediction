# AI-Powered Distillation Intelligence

### Machine Learning-Based Prediction, Monitoring and Optimization of Distillation Processes

AI-powered Chemical Engineering application for **predicting and analyzing ethanol purity in a distillation column using Machine Learning**.

The application combines **Chemical Engineering process parameters, feature engineering, a trained Random Forest regression model, and an interactive web dashboard** to provide real-time predictions and process analysis.

---

## 🚀 Project Overview

Distillation is one of the most widely used separation processes in the chemical industry. The performance of a distillation column depends on several operating parameters such as:

* Column pressure
* Reflux flow
* Vapor flow
* Feed flow
* Distillate flow
* Bottoms flow
* Temperature

Understanding the relationship between these parameters and product purity can be difficult using conventional calculations alone.

This project uses a **Machine Learning-based prediction system** to estimate the ethanol mole fraction from operating conditions and provide additional process insights.

### Key capabilities

* 🎯 Ethanol mole fraction prediction
* 📊 Interactive process dashboard
* 🔬 Automated feature engineering
* 📈 Feature importance analysis
* 🔄 What-If process analysis
* 📁 Batch prediction using CSV files
* 🤖 DistilBot for process-related questions
* 🩺 Model and API health monitoring
* 📚 Interactive API documentation
* 🧪 Automated model and API testing

---

# 🧠 Machine Learning Model

The prediction system uses a trained:

**Random Forest Regressor**

The model predicts the **ethanol mole fraction** from engineered process features.

The trained ML artifacts used by the application are:

```text
model.pkl
scaler.pkl
features_names.pkl
```

### Model pipeline

```text
Process Operating Parameters
            │
            ▼
     Feature Engineering
            │
            ▼
     21 Engineered Features
            │
            ▼
       Feature Ordering
            │
            ▼
        StandardScaler
            │
            ▼
   Random Forest Regressor
            │
            ▼
   Ethanol Mole Fraction
            │
            ▼
 Purity Classification / Analysis
```

The application uses `features_names.pkl` as the authoritative feature ordering to ensure that the input supplied to the model always matches the feature structure used during model development.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Web Dashboard     │
                    │ HTML / CSS / JS      │
                    └──────────┬──────────┘
                               │
                               │ JSON / HTTP
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │      Backend        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ 21 Process Features │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Feature Ordering  │
                    │ features_names.pkl  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   StandardScaler    │
                    │     scaler.pkl      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Random Forest Model │
                    │     model.pkl       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Prediction &        │
                    │ Process Analytics   │
                    └─────────────────────┘
```

---

# 🛠️ Technology Stack

## Backend

* Python
* FastAPI
* Uvicorn
* Pydantic
* Scikit-learn
* NumPy
* Pandas

## Frontend

* HTML5
* CSS3
* Vanilla JavaScript
* Chart.js

## Machine Learning

* Random Forest Regression
* StandardScaler
* Feature Engineering
* Feature Importance Analysis

## AI Assistant

* Groq API
* Rule-based fallback assistant

## Testing

* Pytest
* Model verification scripts
* API testing

---

# 📁 Project Structure

```text
distillation-intelligence-app/
│
├── backend/
│   ├── main.py
│   ├── model_service.py
│   ├── schemas.py
│   ├── chat_service.py
│   │
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── features_names.pkl
│   ├── feature_importance.png
│   │
│   └── tests/
│       └── test_prediction.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── scripts/
│   └── verify_feature_order.py
│
├── run.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/distillation-intelligence-app.git
cd distillation-intelligence-app
```

---

## 2. Create a virtual environment

### Windows

Python 3.12 is recommended for the project environment.

```powershell
py -3.12 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

The application can run without an external AI API key because DistilBot includes a rule-based fallback.

For AI-powered responses using Groq, create a `.env` file from the provided template:

```powershell
copy .env.example .env
```

Then add:

```env
GROQ_API_KEY=your_api_key_here
```

### Important

Never commit your `.env` file or API keys to GitHub.

The `.gitignore` file should exclude:

```text
.env
.venv/
__pycache__/
```

---

# ▶️ Running the Application

Start the application with:

```powershell
python run.py
```

The application will be available at:

```text
http://127.0.0.1:8000
```

The FastAPI interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

# 🧪 Model Verification

The project includes a standalone verification script for checking the feature ordering used by the ML pipeline.

Run:

```powershell
python scripts\verify_feature_order.py
```

The verification confirms that the engineered feature vector follows the authoritative ordering stored in:

```text
backend/features_names.pkl
```

The current verified prediction for the default example is approximately:

```text
0.8246
```

or:

```text
82.46% ethanol mole fraction
```

---

# 📊 Process Inputs

The model receives seven primary process inputs.

| Parameter |      Range | Default | Unit    |
| --------- | ---------: | ------: | ------- |
| Pressure  |  0.5 – 3.0 |    1.01 | bar     |
| T1        |   60 – 120 |    78.0 | °C      |
| L         | 300 – 1200 |   780.0 | kmol/hr |
| V         | 600 – 1500 |  1040.0 | kmol/hr |
| D         |  100 – 500 |   260.0 | kmol/hr |
| B         |  100 – 500 |   340.0 | kmol/hr |
| F         |  350 – 700 |   580.0 | kmol/hr |

These raw process parameters are transformed into engineered features before being passed to the trained model.

---

# 🔬 Feature Engineering

The seven raw process variables are transformed into **21 model features**.

The feature set includes:

```text
Pressure
L
V
D
B
F
Temp_Bottom
Reflux_Ratio
Reboiler_Intensity
Condenser_Load
Distillate_Withdrawal
Bottoms_Withdrawal
Column_Load
Feed_Normalized
Reflux_x_Temp_Top
Reflux_x_Temp_Diff
Reboiler_x_Temp_Bottom
Feed_x_Reflux
Feed_x_Reboiler
Separation_Duty
Column_Efficiency
```

This allows the model to capture relationships between the basic operating conditions and derived process characteristics.

---

# 🎯 Prediction Output

The prediction system provides:

* Predicted ethanol mole fraction
* Purity percentage
* Process status
* Mass balance information
* Derived process metrics
* Model-based process analysis

The application classifies the predicted purity using the following thresholds:

| Predicted Mole Fraction | Status     |
| ----------------------: | ---------- |
|                  ≥ 0.82 | OPTIMAL    |
|             0.75 – 0.82 | ACCEPTABLE |
|                  < 0.75 | BELOW SPEC |

---

# 🔄 What-If Analysis

The What-If Analysis module allows users to modify process conditions and compare predictions.

Example:

```text
Current Conditions
        │
        ▼
Prediction A
        │
        │
Change Reflux Flow
        │
        ▼
Prediction B
        │
        ▼
Prediction Difference
```

This can be used to explore how changes in operating conditions influence predicted product purity.

---

# 📁 Batch Prediction

The application supports prediction for multiple process conditions through CSV upload.

Required columns:

```text
pressure
t1
L
V
D
B
F
```

Example:

```csv
pressure,t1,L,V,D,B,F
1.01,78,780,1040,260,340,580
1.20,82,820,1100,270,350,600
```

The application generates predictions for each row and provides the results for further analysis.

---

# 📈 Feature Importance

The application provides feature importance information from the trained Random Forest model.

This helps identify which engineered process variables have the greatest influence on the model's predictions.

Feature importance can be accessed through:

```text
/api/feature-importance
```

---

# 🤖 DistilBot

**DistilBot** is the project's AI-assisted process information module.

It can help users understand:

* Distillation concepts
* Process parameters
* Prediction results
* Purity interpretation
* Operating conditions
* General process-related questions

The system supports two modes:

```text
GROQ_API_KEY available
        │
        ▼
   Groq-powered AI
```

or:

```text
No API key
        │
        ▼
Rule-based fallback
```

The application therefore remains functional even when an external AI API is not configured.

---

# 🔌 API Endpoints

| Method | Endpoint                      | Description                              |
| ------ | ----------------------------- | ---------------------------------------- |
| GET    | `/api/health`                 | Checks model and artifact status         |
| GET    | `/api/model-info`             | Returns model information                |
| GET    | `/api/feature-importance`     | Returns model feature importance         |
| GET    | `/api/input-ranges`           | Returns process input ranges             |
| POST   | `/api/predict`                | Generates a single prediction            |
| POST   | `/api/what-if`                | Compares current and modified conditions |
| POST   | `/api/batch-predict`          | Performs CSV batch prediction            |
| GET    | `/api/batch-predict/template` | Downloads CSV template                   |
| POST   | `/api/chat`                   | Communicates with DistilBot              |

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 📡 Example Prediction Request

### Endpoint

```text
POST /api/predict
```

### Request

```json
{
  "pressure": 1.01,
  "t1": 78.0,
  "L": 780.0,
  "V": 1040.0,
  "D": 260.0,
  "B": 340.0,
  "F": 580.0
}
```

### Example Response

```json
{
  "prediction": 0.8246,
  "percent": 82.46,
  "status": "OPTIMAL",
  "status_class": "good"
}
```

Additional derived process information is also returned by the API.

---

# 🧪 Testing

Run the complete test suite using:

```powershell
pytest backend\tests\test_prediction.py -v
```

The tests cover:

* Model artifact loading
* Scaler loading
* Feature ordering
* Prediction functionality
* Feature importance
* Health endpoint
* Prediction endpoint
* What-If analysis
* Batch prediction
* Batch validation
* Chat endpoint

---

# 💡 Example Workflow

A typical user workflow is:

```text
1. Enter distillation operating conditions
             ↓
2. Submit prediction request
             ↓
3. Feature engineering
             ↓
4. Feature scaling
             ↓
5. Random Forest prediction
             ↓
6. Ethanol purity prediction
             ↓
7. Process status
             ↓
8. Feature importance / What-If analysis
```

---

# 🎓 Chemical Engineering Relevance

This project demonstrates the application of **Artificial Intelligence and Machine Learning to Chemical Engineering process systems**.

The project connects:

* Distillation processes
* Process variables
* Feature engineering
* Machine Learning
* Process monitoring
* Predictive modeling
* Data-driven process analysis

It demonstrates how historical process data can be used to build predictive tools that support analysis of chemical process performance.

---

# 🔮 Future Improvements

Potential future extensions include:

* Real-time process data integration
* Historical trend visualization
* Advanced process optimization
* Additional ML models
* Model performance comparison
* Explainable AI techniques
* Anomaly detection
* Experimental/industrial process data integration
* Cloud deployment
* Real-time monitoring
* Integration with process control systems

---

# 👨‍💻 Project

**AI-Powered Distillation Intelligence**

A Chemical Engineering + Artificial Intelligence project focused on applying Machine Learning to distillation process prediction and analysis.

### Core technologies

```text
Python
FastAPI
Scikit-learn
Random Forest
HTML
CSS
JavaScript
Chart.js
Pandas
NumPy
Groq API
Pytest
```

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub.
