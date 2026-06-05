"""
Churn Prediction — Inference Module
=====================================
Loads the trained model and makes predictions on new customer data.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def load_model_artifacts(models_dir: str = MODELS_DIR) -> tuple:
    """Load the saved model, scaler, and feature names."""
    models_dir = os.path.abspath(models_dir)

    model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_names.json")) as f:
        feature_names = json.load(f)

    return model, scaler, feature_names


def predict_single(customer: dict, model=None, scaler=None, feature_names=None) -> dict:
    """
    Predict churn probability for a single customer.

    Args:
        customer: Dict with customer features.
        model, scaler, feature_names: Pre-loaded artifacts (loads from disk if None).

    Returns:
        Dict with 'churn_probability', 'prediction', 'risk_level'.
    """
    if model is None:
        model, scaler, feature_names = load_model_artifacts()

    df = pd.DataFrame([customer])
    return predict_batch(df, model, scaler, feature_names).iloc[0].to_dict()


def predict_batch(df: pd.DataFrame, model=None, scaler=None, feature_names=None) -> pd.DataFrame:
    """
    Predict churn for a batch of customers.

    Args:
        df: DataFrame with customer features.
        model, scaler, feature_names: Pre-loaded artifacts (loads from disk if None).

    Returns:
        DataFrame with added 'churn_probability', 'prediction', 'risk_level' columns.
    """
    if model is None:
        model, scaler, feature_names = load_model_artifacts()

    result = df.copy()

    # Ensure all required features exist, fill missing with 0
    for col in feature_names:
        if col not in result.columns:
            result[col] = 0

    X = result[feature_names].values
    X_scaled = scaler.transform(X)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result["churn_probability"] = np.round(probabilities, 4)
    result["prediction"] = predictions
    result["risk_level"] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    return result


def prepare_customer_for_prediction(raw_input: dict, feature_names: list) -> dict:
    """
    Transform raw customer input (human-readable) into model features.
    Handles one-hot encoding and numeric conversions.
    """
    processed = {}

    # Numeric features pass through
    numeric_keys = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
        "num_services", "has_premium_support", "charge_per_service",
        "avg_monthly_charge", "is_new_month_to_month", "is_long_term_customer",
    ]
    for key in numeric_keys:
        if key in raw_input:
            processed[key] = raw_input[key]

    # Binary encoded fields
    binary_fields = {
        "Partner": "Partner_enc",
        "Dependents": "Dependents_enc",
        "PhoneService": "PhoneService_enc",
        "PaperlessBilling": "PaperlessBilling_enc",
    }
    for field, enc_name in binary_fields.items():
        if field in raw_input:
            processed[enc_name] = 1 if raw_input[field] == "Yes" else 0

    # Gender
    if "gender" in raw_input:
        processed["gender_enc"] = 1 if raw_input["gender"] == "Male" else 0

    # One-hot fields
    onehot_fields = {
        "Contract": ["Contract_Month-to-month", "Contract_One year", "Contract_Two year"],
        "InternetService": ["InternetService_DSL", "InternetService_Fiber optic", "InternetService_No"],
        "PaymentMethod": [
            "PaymentMethod_Bank transfer (automatic)",
            "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check",
        ],
    }
    for field, columns in onehot_fields.items():
        if field in raw_input:
            for col in columns:
                val_name = col.split("_", 1)[1]
                processed[col] = 1 if raw_input[field] == val_name else 0

    # Service one-hot fields
    service_fields = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for svc in service_fields:
        if svc in raw_input:
            possible_vals = ["Yes", "No", "No internet service"]
            if svc == "MultipleLines":
                possible_vals = ["Yes", "No", "No phone service"]
            for val in possible_vals:
                col = f"{svc}_{val}"
                processed[col] = 1 if raw_input[svc] == val else 0

    # Fill any missing feature with 0
    for feat in feature_names:
        if feat not in processed:
            processed[feat] = 0

    return processed
