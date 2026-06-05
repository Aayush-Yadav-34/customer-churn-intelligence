"""
Data Cleaning Pipeline
======================
Handles missing values, duplicates, anomalies, type casting, and validation.
Outputs cleaned data to data/processed/cleaned_churn.csv.
"""

import os
import sys
import numpy as np
import pandas as pd

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "telco_churn.csv")
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_churn.csv")


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV file."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print(f"❌ Raw data not found at {path}")
        print("   Run `python src/generate_dataset.py` first.")
        sys.exit(1)
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Detect and impute missing / blank values in TotalCharges."""
    report = {}

    # TotalCharges has blank strings in the raw data
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    blank_mask = df["TotalCharges"] == ""
    report["total_charges_blank"] = int(blank_mask.sum())

    # Convert to numeric, blanks become NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Impute: tenure * MonthlyCharges (logical approximation)
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = (
        df.loc[df["TotalCharges"].isna(), "tenure"]
        * df.loc[df["TotalCharges"].isna(), "MonthlyCharges"]
    )

    # Check for any other missing values
    other_nulls = df.isnull().sum()
    other_nulls = other_nulls[other_nulls > 0]
    report["other_nulls"] = other_nulls.to_dict() if len(other_nulls) > 0 else {}

    return df, report


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows based on customerID."""
    initial_count = len(df)

    # First try exact duplicates on customerID
    df = df.drop_duplicates(subset=["customerID"], keep="first")

    # Also remove full-row duplicates (different customerID but same data unlikely)
    df = df.drop_duplicates(keep="first")

    removed = initial_count - len(df)
    return df, removed


def detect_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Flag anomalous values using IQR method on numerical columns."""
    report = {}
    numerical_cols = ["MonthlyCharges", "TotalCharges", "tenure"]

    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        report[col] = {
            "outliers_found": int(outliers),
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
        }

    # Cap MonthlyCharges anomalies (the 150-250 range ones we injected)
    mc_upper = df["MonthlyCharges"].quantile(0.75) + 1.5 * (
        df["MonthlyCharges"].quantile(0.75) - df["MonthlyCharges"].quantile(0.25)
    )
    df["MonthlyCharges"] = df["MonthlyCharges"].clip(upper=mc_upper)

    return df, report


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types."""
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # Encode Churn as binary
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    return df


def validate(df: pd.DataFrame) -> bool:
    """Run validation checks on the cleaned data."""
    checks = []

    # No nulls
    null_count = df.isnull().sum().sum()
    checks.append(("No null values", null_count == 0, f"{null_count} nulls found"))

    # Expected row count range
    checks.append((
        "Row count in range [6500, 7500]",
        6500 <= len(df) <= 7500,
        f"Got {len(df)} rows",
    ))

    # Expected columns
    expected = {
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    }
    missing_cols = expected - set(df.columns)
    checks.append((
        "All expected columns present",
        len(missing_cols) == 0,
        f"Missing: {missing_cols}",
    ))

    # MonthlyCharges positive
    checks.append((
        "MonthlyCharges all positive",
        (df["MonthlyCharges"] > 0).all(),
        "Some non-positive values found",
    ))

    # Tenure non-negative
    checks.append((
        "Tenure non-negative",
        (df["tenure"] >= 0).all(),
        "Some negative values found",
    ))

    all_passed = True
    print("\n── Validation Checks ──")
    for name, passed, detail in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}" + (f" — {detail}" if not passed else ""))
        if not passed:
            all_passed = False

    return all_passed


def run_pipeline(raw_path: str = RAW_PATH, clean_path: str = CLEAN_PATH) -> pd.DataFrame:
    """Execute the full cleaning pipeline."""
    print("=" * 60)
    print("  DATA CLEANING PIPELINE")
    print("=" * 60)

    # Load
    df = load_raw_data(raw_path)
    print(f"\n📥 Loaded raw data: {len(df)} rows, {len(df.columns)} columns")

    # Clean missing values
    df, missing_report = handle_missing_values(df)
    print(f"\n🔧 Missing values:")
    print(f"   TotalCharges blanks imputed: {missing_report['total_charges_blank']}")
    if missing_report["other_nulls"]:
        for col, cnt in missing_report["other_nulls"].items():
            print(f"   {col}: {cnt} nulls remaining")

    # Remove duplicates
    df, dups_removed = remove_duplicates(df)
    print(f"\n🗑️  Duplicates removed: {dups_removed}")
    print(f"   Rows remaining: {len(df)}")

    # Detect anomalies
    df, anomaly_report = detect_anomalies(df)
    print(f"\n🔍 Anomaly detection (IQR method):")
    for col, info in anomaly_report.items():
        print(f"   {col}: {info['outliers_found']} outliers "
              f"(bounds: [{info['lower_bound']}, {info['upper_bound']}])")

    # Type casting
    df = cast_types(df)
    print("\n🏷️  Types cast (SeniorCitizen→int, TotalCharges→float, Churn_Binary added)")

    # Validate
    valid = validate(df)

    # Save
    clean_path = os.path.abspath(clean_path)
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)
    print(f"\n💾 Cleaned data saved to: {clean_path}")
    print(f"   Final shape: {df.shape}")

    if not valid:
        print("\n⚠️  Some validation checks failed — review output above.")

    return df


if __name__ == "__main__":
    run_pipeline()
