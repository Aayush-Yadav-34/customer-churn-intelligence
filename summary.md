# Customer Churn Intelligence Platform — Project Summary

## What This Project Does

This is a **Python-based end-to-end analytics platform** that analyzes customer behavior, identifies why customers leave (churn), predicts which customers are likely to churn using machine learning, and presents everything through an interactive dashboard and automated PDF reports.

It targets businesses like telecom companies, SaaS platforms, and subscription services that want to understand and reduce customer attrition.

---

## Folder Structure

```
customer-churn-intelligence/
│
├── .streamlit/          → Streamlit dashboard configuration
├── data/                → All datasets (raw and processed)
│   ├── raw/             → Original unprocessed data
│   └── processed/       → Cleaned and feature-engineered data
├── src/                 → Core Python pipeline scripts
├── dashboard/           → Interactive Streamlit web application
├── models/              → Saved ML model artifacts
├── reports/             → Generated charts, insights, and PDF report
├── requirements.txt     → Python dependencies
├── run_pipeline.py      → End-to-end pipeline runner
├── README.md            → Project documentation
└── summary.md           → This file
```

---

## File-by-File Breakdown

### Root Files

| File | Description |
|---|---|
| `requirements.txt` | Lists all 12 Python packages needed to run the project (pandas, scikit-learn, xgboost, shap, streamlit, reportlab, etc.) |
| `run_pipeline.py` | Runs the entire end-to-end pipeline (generation to report generation) with a single command |
| `README.md` | Full project documentation with setup instructions, features list, and tech stack badges |
| `summary.md` | This summary document |

---

### `.streamlit/`

| File | Description |
|---|---|
| `config.toml` | Defines the dashboard's dark theme — deep navy background, violet accent colors (#6C63FF), and sans-serif font |

---

### `data/raw/`

| File | Description |
|---|---|
| `telco_churn.csv` | Synthetic dataset with 7,058 rows and 21 columns. Contains customer demographics, services subscribed, billing info, and churn status. Includes intentional noise (missing values, duplicates, outliers) for the cleaning pipeline to process |

### `data/processed/`

| File | Description |
|---|---|
| `cleaned_churn.csv` | Output of the cleaning pipeline — 7,043 rows with no nulls, no duplicates, capped outliers, and a new `Churn_Binary` column |
| `features_churn.csv` | Output of feature engineering — 55 columns including tenure bins, service counts, charge ratios, and one-hot encoded categoricals |
| `segmented_churn.csv` | Cleaned data with K-Means cluster labels (`Segment` column) added for each customer |

---

### `src/` — Core Pipeline

These scripts are designed to be run **in order** (each depends on the previous step's output):

| # | File | What It Does |
|---|---|---|
| 1 | `generate_dataset.py` | Creates a realistic synthetic Telco customer churn dataset. Generates 7,043 customers with proper correlations (e.g., month-to-month contracts → higher churn) and injects noise (11 missing values, 15 duplicates, 5 extreme charges) for testing the cleaning pipeline |
| 2 | `data_cleaning.py` | Cleans the raw data: converts blank `TotalCharges` to numeric and imputes using `tenure × MonthlyCharges`, removes duplicate rows by `customerID`, caps extreme outliers using IQR method, casts data types, and runs 5 validation checks |
| 3 | `feature_engineering.py` | Creates 33 new features from cleaned data: `tenure_group` (5 bins), `avg_monthly_charge`, `num_services` (count of subscribed services), `has_premium_support` (binary flag), `charge_per_service`, `is_new_month_to_month` (interaction), and one-hot encodes all categorical columns |
| 4 | `eda.py` | Generates 8 dark-themed analysis charts: churn distribution (pie + bar), revenue loss analysis, demographics vs churn, contract type breakdown, tenure analysis, payment method impact, service usage heatmap, and numerical correlation matrix |
| 5 | `segmentation.py` | Groups customers using K-Means clustering on tenure, charges, and service count. Uses elbow method + silhouette score to find optimal K. Assigns meaningful names (Loyal, High-Value, New Customers) based on cluster centroids. Generates PCA scatter plot |
| 6 | `train_model.py` | Trains 3 ML models with hyperparameter tuning (GridSearchCV, 5-fold CV): Logistic Regression, Random Forest, and XGBoost. Applies SMOTE for class imbalance. Evaluates each on Accuracy, Precision, Recall, F1, ROC-AUC. Auto-selects and saves the best model |
| 7 | `prediction.py` | Inference module used by the dashboard. Loads the saved model and scaler, accepts single customer dicts or batch DataFrames, returns churn probability and risk level (Low/Medium/High). Handles feature transformation from raw inputs |
| 8 | `explainability.py` | Uses SHAP (SHapley Additive exPlanations) to explain model predictions. Generates feature importance bar chart, beeswarm summary plot, and dependence plots. Produces natural-language insights like "Month-to-month customers are 2.5x more likely to churn" |
| 9 | `report_generator.py` | Builds a professional multi-page PDF report using ReportLab. Includes cover page, KPI table, embedded EDA charts, segmentation profiles, model comparison table, SHAP feature importance, and 6 prioritized business recommendations |

---

### `dashboard/`

| File | Description |
|---|---|
| `app.py` | **Interactive Streamlit web application** with 4 pages: |

**Page 1 — Executive Summary:** KPI cards (total customers, churn rate, revenue at risk, avg tenure), churn donut chart, revenue breakdown bar chart, segment distribution, top churn drivers list.

**Page 2 — Customer Insights:** Interactive Plotly charts with filters (gender, contract, internet service). Demographics breakdown, contract analysis, monthly charges histogram, tenure scatter plot, service correlation heatmap, and a segment explorer with drill-down.

**Page 3 — Prediction Center:** Two modes — (1) Manual: fill a form with sliders/dropdowns for customer features → get churn probability with a gauge visualization and risk badge. (2) Batch: upload a CSV → predict all rows → download results.

**Page 4 — Business Recommendations:** Lists all SHAP-derived insights, 6 actionable retention strategies with priority levels (High/Medium/Low), cost-benefit estimation, and a button to generate and download the PDF report.

---

### `models/`

| File | Description |
|---|---|
| `best_model.pkl` | Serialized best-performing ML model (Logistic Regression). Loaded by the dashboard for predictions |
| `scaler.pkl` | StandardScaler fitted on training data. Required to normalize new inputs before prediction |
| `feature_names.json` | Ordered list of 46 feature column names. Ensures prediction inputs match the training schema exactly |
| `model_comparison.json` | Performance metrics (accuracy, precision, recall, F1, AUC) for all 3 trained models |

---

### `reports/`

| File | Description |
|---|---|
| `01_churn_distribution.png` | Pie chart + bar chart showing overall churn split |
| `02_revenue_analysis.png` | Revenue comparison between retained and churned customers |
| `03_demographics_churn.png` | Gender, senior citizen, partner, dependents vs churn rates |
| `04_contract_churn.png` | Stacked bar chart of contract types vs churn |
| `05_tenure_analysis.png` | Tenure distribution and box plot by churn status |
| `06_payment_method_churn.png` | Payment method churn rates (horizontal bar) |
| `07_services_heatmap.png` | Correlation heatmap of services and churn |
| `08_correlation_matrix.png` | Numerical features correlation matrix |
| `09_optimal_k.png` | Elbow method + silhouette score for K selection |
| `10_customer_segments.png` | PCA scatter plot of customer segments |
| `11_model_comparison.png` | Bar chart comparing 3 model metrics |
| `12_roc_curves.png` | ROC curves for all models |
| `13_confusion_matrices.png` | Confusion matrices side by side |
| `14_feature_importance.png` | Top 15 SHAP feature importance (bar) |
| `15_shap_summary.png` | SHAP beeswarm summary plot |
| `16_shap_dependence.png` | SHAP dependence plots for top 3 features |
| `segment_profiles.json` | Segment statistics (count, avg tenure, avg charges, churn rate) |
| `insights.json` | Natural-language business insights with priority levels |
| `churn_report.pdf` | Full auto-generated business report (multi-page PDF) |

---

## How to Run This Project

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the Data Pipeline
Run the entire end-to-end pipeline with a single command from the project root:

```bash
python run_pipeline.py
```

This runs all 8 pipeline steps in sequence, handles Windows Unicode encoding automatically, and generates all cleaned data, models, reports, and insights.

Alternatively, you can run individual scripts from the `src/` directory in order:

```bash
# Generate the synthetic dataset
python src/generate_dataset.py

# Clean the data (handles missing values, duplicates, outliers)
python src/data_cleaning.py

# Create new features for ML
python src/feature_engineering.py

# Generate EDA visualizations
python src/eda.py

# Segment customers using K-Means
python src/segmentation.py

# Train and evaluate ML models
python src/train_model.py

# Generate SHAP explanations and insights
python src/explainability.py

# Generate the PDF business report
python src/report_generator.py
```

### Step 3 — Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
Opens at **http://localhost:8501** with all 4 pages and a synchronized top navigation bar ready to explore.

---

## Key Results

| Metric | Value |
|---|---|
| Total Customers | 7,043 |
| Churn Rate | 37.2% |
| Revenue at Risk | $3,871,929 |
| Best Model | Logistic Regression |
| Best F1 Score | 0.6168 |
| Best ROC-AUC | 0.7492 |
| Customer Segments | 3 (New, High-Value, Loyal) |
| Highest Risk Segment | New Customers (51% churn) |

## Top Insights Discovered
1. **Month-to-month contract** customers are **2.5x** more likely to churn
2. **Short tenure + high charges** (< 12 months, > $70/mo) → **56% churn rate**
3. **Fiber optic** users churn at 47% vs 36% for DSL
4. **Electronic check** users churn at 41% vs 36% for auto-payment
5. Top model drivers: `tenure`, `Contract_Month-to-month`, `Contract_Two year`
