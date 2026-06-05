# 🔍 Customer Churn Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189FDD?style=for-the-badge)](https://xgboost.readthedocs.io)

An end-to-end analytics platform that analyzes customer behavior, identifies churn risk factors, predicts customer attrition using machine learning, and generates actionable business insights through an interactive dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Data Cleaning Pipeline** | Handles missing values, duplicates, anomalies, type casting, and validation |
| **Exploratory Data Analysis** | 8+ publication-quality visualizations covering demographics, services, revenue, and tenure |
| **Customer Segmentation** | K-Means clustering into 4 segments: Loyal, High-Value, At-Risk, New Customers |
| **Churn Prediction Engine** | Logistic Regression, Random Forest, XGBoost with GridSearchCV and SMOTE |
| **Explainable AI** | SHAP feature importance, beeswarm plots, dependence plots, natural-language insights |
| **Interactive Dashboard** | 4-page Streamlit app with KPIs, filters, prediction center, and batch upload |
| **Automated PDF Report** | Professional business report with KPIs, charts, findings, and recommendations |

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Data Processing** | Python, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-Learn, XGBoost, SHAP, imbalanced-learn |
| **Dashboard** | Streamlit |
| **Reporting** | ReportLab |

---

## 📁 Project Structure

```
customer-churn-intelligence/
├── data/
│   ├── raw/                    # Raw synthetic dataset
│   └── processed/              # Cleaned & feature-engineered data
├── src/
│   ├── generate_dataset.py     # Synthetic data generation
│   ├── data_cleaning.py        # Data cleaning pipeline
│   ├── feature_engineering.py  # Feature derivation & encoding
│   ├── eda.py                  # Exploratory data analysis
│   ├── segmentation.py         # K-Means customer segmentation
│   ├── train_model.py          # ML model training & evaluation
│   ├── prediction.py           # Inference module
│   ├── explainability.py       # SHAP analysis
│   └── report_generator.py     # PDF report generation
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── models/                     # Saved model artifacts
├── reports/                    # Generated charts & reports
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Aayush-Yadav-34/customer-churn-intelligence.git
cd customer-churn-intelligence

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

> **Note:** The `data/`, `models/`, and `reports/` folders are empty after cloning. Running the pipeline below generates all datasets, model artifacts, charts, and reports.

```bash
# 1. Generate synthetic dataset
python src/generate_dataset.py

# 2. Clean the data
python src/data_cleaning.py

# 3. Engineer features
python src/feature_engineering.py

# 4. Run exploratory data analysis
python src/eda.py

# 5. Segment customers
python src/segmentation.py

# 6. Train ML models
python src/train_model.py

# 7. Generate SHAP explanations
python src/explainability.py

# 8. Generate PDF report
python src/report_generator.py
```

### Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`.

---

## 📊 Dashboard Pages

### 1. Executive Summary
KPI cards, churn distribution donut chart, revenue breakdown, customer segments, and top churn drivers.

### 2. Customer Insights
Interactive charts with filters for demographics, contract types, monthly charges, tenure, and service usage heatmap. Includes a segment explorer with drill-down capability.

### 3. Prediction Center
- **Single Customer**: Form with sliders and dropdowns → churn probability gauge
- **Batch Upload**: Upload CSV → predict all → download results

### 4. Business Recommendations
Data-driven retention strategies with priority levels, cost-benefit estimates, and one-click PDF report generation.

---

## 🤖 Model Performance

The platform trains and compares three models:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| XGBoost | - | - | - | - | - |

*Metrics populated after training.*

---

## 💡 Key Insights (Examples)

- 🔴 Customers on month-to-month contracts are **3x more likely** to churn
- 🔴 Short tenure (<12 months) + high charges (>$70) → **65% churn rate**
- 🟡 Fiber optic customers churn more despite paying premium prices
- 🟡 Electronic check users have the highest churn rate among payment methods
- 🔵 Online Security and Tech Support are strong retention drivers

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
