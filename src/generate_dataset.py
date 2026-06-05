"""
Synthetic Telco Customer Churn Dataset Generator
=================================================
Generates a realistic dataset matching the Kaggle Telco Customer Churn schema.
- 7,043 base customers + 15 duplicate rows for testing
- Realistic distributions and correlations
- Intentional noise: missing TotalCharges, duplicates, anomalous charges
"""

import os
import uuid
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────

NUM_CUSTOMERS = 7043
NUM_DUPLICATES = 15
NUM_MISSING_TOTAL_CHARGES = 11
RANDOM_SEED = 42
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "telco_churn.csv")


def _generate_customer_ids(n: int) -> list[str]:
    """Generate unique customer IDs like '7590-VHVEG'."""
    rng = np.random.default_rng(RANDOM_SEED)
    ids = []
    for _ in range(n):
        prefix = str(rng.integers(1000, 9999))
        suffix = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), size=5))
        ids.append(f"{prefix}-{suffix}")
    return ids


def generate_dataset() -> pd.DataFrame:
    """Generate the full synthetic churn dataset."""
    rng = np.random.default_rng(RANDOM_SEED)

    n = NUM_CUSTOMERS
    customer_ids = _generate_customer_ids(n)

    # ── Demographics ───────────────────────────────────────────────────────
    gender = rng.choice(["Male", "Female"], size=n, p=[0.505, 0.495])
    senior_citizen = rng.choice([0, 1], size=n, p=[0.838, 0.162])
    partner = rng.choice(["Yes", "No"], size=n, p=[0.484, 0.516])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.299, 0.701])

    # ── Tenure (months) — bimodal distribution ─────────────────────────────
    tenure = np.zeros(n, dtype=int)
    mask_new = rng.random(n) < 0.35
    tenure[mask_new] = rng.integers(1, 13, size=mask_new.sum())
    tenure[~mask_new] = rng.integers(1, 73, size=(~mask_new).sum())

    # ── Contract type (correlated with tenure) ─────────────────────────────
    contract = np.empty(n, dtype=object)
    for i in range(n):
        if tenure[i] <= 12:
            contract[i] = rng.choice(
                ["Month-to-month", "One year", "Two year"], p=[0.75, 0.18, 0.07]
            )
        elif tenure[i] <= 36:
            contract[i] = rng.choice(
                ["Month-to-month", "One year", "Two year"], p=[0.45, 0.35, 0.20]
            )
        else:
            contract[i] = rng.choice(
                ["Month-to-month", "One year", "Two year"], p=[0.25, 0.30, 0.45]
            )

    # ── Services ───────────────────────────────────────────────────────────
    phone_service = rng.choice(["Yes", "No"], size=n, p=[0.903, 0.097])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], size=n, p=[0.422, 0.578]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n, p=[0.344, 0.440, 0.216]
    )

    # Internet-dependent services
    internet_dependent = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    services = {}
    for svc in internet_dependent:
        svc_values = np.empty(n, dtype=object)
        for i in range(n):
            if internet_service[i] == "No":
                svc_values[i] = "No internet service"
            else:
                # Fiber optic customers tend to have fewer add-ons
                if internet_service[i] == "Fiber optic":
                    svc_values[i] = rng.choice(["Yes", "No"], p=[0.35, 0.65])
                else:
                    svc_values[i] = rng.choice(["Yes", "No"], p=[0.50, 0.50])
        services[svc] = svc_values

    # ── Billing ────────────────────────────────────────────────────────────
    paperless_billing = rng.choice(["Yes", "No"], size=n, p=[0.593, 0.407])
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n,
        p=[0.336, 0.228, 0.219, 0.217],
    )

    # ── Monthly Charges (correlated with services) ─────────────────────────
    monthly_charges = np.zeros(n)
    for i in range(n):
        base = 19.0
        if internet_service[i] == "DSL":
            base += rng.uniform(20, 35)
        elif internet_service[i] == "Fiber optic":
            base += rng.uniform(35, 55)
        if phone_service[i] == "Yes":
            base += rng.uniform(0, 10)
        for svc in internet_dependent:
            if services[svc][i] == "Yes":
                base += rng.uniform(5, 15)
        monthly_charges[i] = round(base + rng.normal(0, 3), 2)

    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

    # ── Total Charges ──────────────────────────────────────────────────────
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, size=n), 2)
    total_charges = np.clip(total_charges, 18.8, 8700.0)

    # ── Churn (correlated with contract, tenure, charges) ──────────────────
    churn_prob = np.full(n, 0.15)  # Base probability

    # Contract effect
    churn_prob = np.where(
        np.array(contract) == "Month-to-month", churn_prob + 0.25, churn_prob
    )
    churn_prob = np.where(
        np.array(contract) == "Two year", churn_prob - 0.10, churn_prob
    )

    # Tenure effect
    churn_prob = np.where(tenure <= 6, churn_prob + 0.15, churn_prob)
    churn_prob = np.where(tenure > 48, churn_prob - 0.10, churn_prob)

    # Charges effect
    median_charge = np.median(monthly_charges)
    churn_prob = np.where(
        monthly_charges > median_charge + 20, churn_prob + 0.10, churn_prob
    )

    # Internet service effect
    churn_prob = np.where(
        np.array(internet_service) == "Fiber optic", churn_prob + 0.10, churn_prob
    )
    churn_prob = np.where(
        np.array(internet_service) == "No", churn_prob - 0.10, churn_prob
    )

    # Payment method effect
    churn_prob = np.where(
        np.array(payment_method) == "Electronic check", churn_prob + 0.08, churn_prob
    )

    # Senior citizen effect
    churn_prob = np.where(senior_citizen == 1, churn_prob + 0.05, churn_prob)

    # No online security / tech support effect
    churn_prob = np.where(
        services["OnlineSecurity"] == "No", churn_prob + 0.05, churn_prob
    )
    churn_prob = np.where(
        services["TechSupport"] == "No", churn_prob + 0.03, churn_prob
    )

    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = np.where(rng.random(n) < churn_prob, "Yes", "No")

    # ── Build DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": services["OnlineSecurity"],
            "OnlineBackup": services["OnlineBackup"],
            "DeviceProtection": services["DeviceProtection"],
            "TechSupport": services["TechSupport"],
            "StreamingTV": services["StreamingTV"],
            "StreamingMovies": services["StreamingMovies"],
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

    # ── Inject noise ───────────────────────────────────────────────────────

    # 1. Missing TotalCharges (blank strings, as in the real Kaggle dataset)
    missing_idx = rng.choice(
        df[df["tenure"] == 1].index,
        size=min(NUM_MISSING_TOTAL_CHARGES, (df["tenure"] == 1).sum()),
        replace=False,
    )
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[missing_idx, "TotalCharges"] = " "

    # 2. Duplicate rows
    dup_idx = rng.choice(df.index, size=NUM_DUPLICATES, replace=False)
    duplicates = df.loc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # 3. A few anomalous charges (extremely high)
    anomaly_idx = rng.choice(df.index, size=5, replace=False)
    df.loc[anomaly_idx, "MonthlyCharges"] = rng.uniform(150, 250, size=5).round(2)

    return df


def main():
    """Generate and save the dataset."""
    print("=" * 60)
    print("  SYNTHETIC DATASET GENERATOR")
    print("=" * 60)

    df = generate_dataset()

    output_path = os.path.abspath(OUTPUT_PATH)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Dataset saved to: {output_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
    print(f"   Missing TotalCharges: {(df['TotalCharges'].str.strip() == '').sum()}")
    print(f"   Duplicate rows injected: {NUM_DUPLICATES}")
    print(f"   Anomalous charges injected: 5")


if __name__ == "__main__":
    main()
