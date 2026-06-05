"""
Feature Engineering Module
==========================
Derives new features from cleaned data for ML modeling.
Outputs feature-engineered data to data/processed/features_churn.csv.
"""

import os
import sys
import numpy as np
import pandas as pd

CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_churn.csv")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features_churn.csv")


def load_cleaned_data(path: str = CLEAN_PATH) -> pd.DataFrame:
    """Load the cleaned dataset."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print(f"❌ Cleaned data not found at {path}")
        print("   Run `python src/data_cleaning.py` first.")
        sys.exit(1)
    return pd.read_csv(path)


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin tenure into meaningful groups."""
    bins = [0, 12, 24, 48, 60, 72]
    labels = ["0-12", "13-24", "25-48", "49-60", "61-72"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    return df


def add_avg_monthly_charge(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate average monthly charge over the customer's tenure."""
    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["avg_monthly_charge"] = df["avg_monthly_charge"].round(2)
    return df


def add_num_services(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of subscribed services."""
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = 0
    for col in service_cols:
        if col == "InternetService":
            df["num_services"] += (df[col] != "No").astype(int)
        elif col in ("MultipleLines",):
            df["num_services"] += (df[col] == "Yes").astype(int)
        else:
            df["num_services"] += (df[col] == "Yes").astype(int)
    return df


def add_premium_support(df: pd.DataFrame) -> pd.DataFrame:
    """Flag customers with premium support bundle (security + tech support + device protection)."""
    df["has_premium_support"] = (
        (df["OnlineSecurity"] == "Yes") &
        (df["TechSupport"] == "Yes") &
        (df["DeviceProtection"] == "Yes")
    ).astype(int)
    return df


def add_charge_per_service(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate charge per service subscribed."""
    df["charge_per_service"] = (
        df["MonthlyCharges"] / df["num_services"].clip(lower=1)
    ).round(2)
    return df


def add_contract_tenure_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between contract and tenure."""
    df["is_new_month_to_month"] = (
        (df["Contract"] == "Month-to-month") & (df["tenure"] <= 12)
    ).astype(int)
    df["is_long_term_customer"] = (
        (df["Contract"].isin(["One year", "Two year"])) & (df["tenure"] > 36)
    ).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features for ML (keeps originals for dashboard)."""
    # Binary columns → 0/1
    binary_map = {"Yes": 1, "No": 0}
    binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        df[f"{col}_enc"] = df[col].map(binary_map)

    # Multi-class columns → one-hot
    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cols, prefix_sep="_", dtype=int)

    # Gender → binary
    df["gender_enc"] = (df["gender"] == "Male").astype(int)

    return df


def run_feature_engineering(
    clean_path: str = CLEAN_PATH,
    features_path: str = FEATURES_PATH,
) -> pd.DataFrame:
    """Execute the full feature engineering pipeline."""
    print("=" * 60)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    df = load_cleaned_data(clean_path)
    print(f"\n📥 Loaded cleaned data: {df.shape}")

    initial_cols = len(df.columns)

    df = add_tenure_group(df)
    df = add_avg_monthly_charge(df)
    df = add_num_services(df)
    df = add_premium_support(df)
    df = add_charge_per_service(df)
    df = add_contract_tenure_interaction(df)
    df = encode_categoricals(df)

    new_cols = len(df.columns) - initial_cols
    print(f"\n🔨 Features added: {new_cols} new columns")
    print(f"   Total columns: {len(df.columns)}")
    print(f"\n   New features:")
    print(f"   • tenure_group (5 bins)")
    print(f"   • avg_monthly_charge")
    print(f"   • num_services (count)")
    print(f"   • has_premium_support (binary)")
    print(f"   • charge_per_service")
    print(f"   • is_new_month_to_month (interaction)")
    print(f"   • is_long_term_customer (interaction)")
    print(f"   • Encoded categoricals (one-hot)")

    features_path = os.path.abspath(features_path)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    df.to_csv(features_path, index=False)
    print(f"\n💾 Feature-engineered data saved to: {features_path}")

    return df


if __name__ == "__main__":
    run_feature_engineering()
