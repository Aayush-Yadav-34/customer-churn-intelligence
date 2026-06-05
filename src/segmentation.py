"""
Customer Segmentation Module
=============================
K-Means clustering to group customers into meaningful segments.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_churn.csv")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG = "#0E1117"
CARD_BG = "#1A1A2E"
TEXT_COLOR = "#E0E0E0"
SEGMENT_COLORS = ["#6C63FF", "#00D4AA", "#FF6B6B", "#FFD93D"]
SEGMENT_NAMES = ["Loyal Customers", "High-Value Customers", "At-Risk Customers", "New Customers"]

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#333355",
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": "#333355",
    "grid.alpha": 0.3,
    "font.size": 11,
})


def load_data(path: str = CLEAN_PATH) -> pd.DataFrame:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print(f"❌ Data not found at {path}")
        sys.exit(1)
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Select and scale features for clustering."""
    feature_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Add num_services if not present
    if "num_services" not in df.columns:
        service_cols = [
            "PhoneService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        df["num_services"] = sum((df[col] == "Yes").astype(int) for col in service_cols)
    feature_cols.append("num_services")

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler, feature_cols


def find_optimal_k(X: np.ndarray, k_range: range = range(2, 8)) -> tuple[int, dict]:
    """Use elbow method + silhouette score to find optimal K."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    # Plot elbow + silhouette
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(k_range), inertias, "o-", color="#6C63FF", linewidth=2, markersize=8)
    axes[0].set_title("Elbow Method", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), silhouettes, "o-", color="#00D4AA", linewidth=2, markersize=8)
    axes[1].set_title("Silhouette Score", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Optimal K Selection", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = os.path.join(os.path.abspath(REPORTS_DIR), "09_optimal_k.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  📊 Saved: 09_optimal_k.png")

    # Select best K (highest silhouette, prefer K=4 as per spec)
    best_k = list(k_range)[np.argmax(silhouettes)]
    # Override to 4 if silhouette is close (within 0.05 of max)
    max_sil = max(silhouettes)
    if 4 in k_range:
        k4_sil = silhouettes[list(k_range).index(4)]
        if max_sil - k4_sil < 0.05:
            best_k = 4

    return best_k, {"inertias": inertias, "silhouettes": silhouettes}


def assign_segment_labels(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Assign meaningful names to cluster labels based on centroids."""
    df = df.copy()
    df["Cluster"] = labels

    # Compute cluster centroids
    centroids = df.groupby("Cluster").agg({
        "tenure": "mean",
        "MonthlyCharges": "mean",
        "TotalCharges": "mean",
    }).round(2)

    # Sort clusters by tenure (ascending) → label assignment
    tenure_order = centroids["tenure"].sort_values().index.tolist()

    # Assignment logic:
    # Lowest tenure → New Customers
    # Highest tenure, lower charges → Loyal Customers
    # High charges → High-Value Customers
    # Remaining → At-Risk Customers
    label_map = {}
    used = set()

    # New Customers: lowest tenure
    label_map[tenure_order[0]] = "New Customers"
    used.add(tenure_order[0])

    # High-Value: highest monthly charges among remaining
    remaining = [c for c in tenure_order if c not in used]
    charges_order = centroids.loc[remaining, "MonthlyCharges"].sort_values(ascending=False)
    label_map[charges_order.index[0]] = "High-Value Customers"
    used.add(charges_order.index[0])

    # Loyal: highest tenure among remaining
    remaining = [c for c in tenure_order if c not in used]
    tenure_remaining = centroids.loc[remaining, "tenure"].sort_values(ascending=False)
    label_map[tenure_remaining.index[0]] = "Loyal Customers"
    used.add(tenure_remaining.index[0])

    # At-Risk: whatever is left
    for c in tenure_order:
        if c not in used:
            label_map[c] = "At-Risk Customers"

    df["Segment"] = df["Cluster"].map(label_map)

    return df, label_map


def plot_segments(df: pd.DataFrame, X_scaled: np.ndarray):
    """2D PCA scatter plot of customer segments."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    segments = df["Segment"].unique()
    color_map = {name: color for name, color in zip(SEGMENT_NAMES, SEGMENT_COLORS)}

    for seg in segments:
        mask = df["Segment"] == seg
        color = color_map.get(seg, "#888888")
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=seg,
                   alpha=0.5, s=20, edgecolors="none")

    ax.set_title("Customer Segments (PCA Projection)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.legend(loc="upper right", fontsize=11, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    out = os.path.join(os.path.abspath(REPORTS_DIR), "10_customer_segments.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  📊 Saved: 10_customer_segments.png")


def run_segmentation(data_path: str = CLEAN_PATH) -> pd.DataFrame:
    """Execute the full segmentation pipeline."""
    print("=" * 60)
    print("  CUSTOMER SEGMENTATION")
    print("=" * 60)

    df = load_data(data_path)
    print(f"\n📥 Loaded data: {df.shape}")

    # Prepare features
    X_scaled, scaler, feature_cols = prepare_features(df)
    print(f"\n🔧 Features used: {feature_cols}")

    # Find optimal K
    print("\n🔍 Finding optimal K...")
    best_k, _ = find_optimal_k(X_scaled)
    print(f"   Optimal K: {best_k}")

    # Fit KMeans
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # Assign labels
    df, label_map = assign_segment_labels(df, labels)
    print(f"\n🏷️  Segment assignments: {label_map}")

    # Plot
    print("\n📊 Generating segment visualizations...")
    plot_segments(df, X_scaled)

    # Segment profiles
    print("\n── Segment Profiles ──")
    profiles = df.groupby("Segment").agg({
        "customerID": "count",
        "tenure": "mean",
        "MonthlyCharges": "mean",
        "TotalCharges": "mean",
        "Churn": lambda x: (x == "Yes").mean(),
    }).round(2)
    profiles.columns = ["Count", "Avg Tenure", "Avg Monthly", "Avg Total", "Churn Rate"]
    print(profiles.to_string())

    # Save profiles
    profiles_path = os.path.join(os.path.abspath(REPORTS_DIR), "segment_profiles.json")
    profiles.reset_index().to_json(profiles_path, orient="records", indent=2)
    print(f"\n💾 Profiles saved to: {profiles_path}")

    # Save segmented data
    seg_path = os.path.join(os.path.abspath(PROCESSED_DIR), "segmented_churn.csv")
    df.to_csv(seg_path, index=False)
    print(f"💾 Segmented data saved to: {seg_path}")

    return df


if __name__ == "__main__":
    run_segmentation()
