"""
Exploratory Data Analysis (EDA) Module
======================================
Generates comprehensive visualizations of customer churn data.
All plots saved to reports/ directory.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_churn.csv")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

# ── Style Configuration ───────────────────────────────────────────────────────

DARK_BG = "#0E1117"
CARD_BG = "#1A1A2E"
TEXT_COLOR = "#E0E0E0"
ACCENT_1 = "#6C63FF"   # Violet
ACCENT_2 = "#00D4AA"   # Teal
ACCENT_3 = "#FF6B6B"   # Coral
ACCENT_4 = "#4ECDC4"   # Mint
PALETTE = [ACCENT_1, ACCENT_3, ACCENT_2, ACCENT_4, "#FFD93D", "#C9B1FF"]

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
    "font.family": "sans-serif",
    "font.size": 11,
})


def load_data(path: str = CLEAN_PATH) -> pd.DataFrame:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print(f"❌ Data not found at {path}")
        sys.exit(1)
    return pd.read_csv(path)


def _save(fig, name: str):
    """Save a figure to the reports directory."""
    out = os.path.join(os.path.abspath(REPORTS_DIR), name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  📊 Saved: {name}")


# ── Plot Functions ─────────────────────────────────────────────────────────────

def plot_churn_distribution(df: pd.DataFrame):
    """Churn distribution — pie chart and bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    churn_counts = df["Churn"].value_counts()
    colors = [ACCENT_2, ACCENT_3]
    axes[0].pie(
        churn_counts, labels=churn_counts.index, autopct="%1.1f%%",
        colors=colors, startangle=90, textprops={"color": TEXT_COLOR, "fontsize": 13},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
    )
    axes[0].set_title("Churn Distribution", fontsize=15, fontweight="bold", pad=15)

    # Bar chart
    bars = axes[1].bar(churn_counts.index, churn_counts.values, color=colors,
                       edgecolor=DARK_BG, linewidth=1.5, width=0.5)
    for bar, val in zip(bars, churn_counts.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     str(val), ha="center", va="bottom", fontsize=13, fontweight="bold")
    axes[1].set_title("Customer Count by Churn Status", fontsize=15, fontweight="bold", pad=15)
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("")

    fig.suptitle("Customer Churn Overview", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "01_churn_distribution.png")


def plot_revenue_analysis(df: pd.DataFrame):
    """Revenue loss analysis — churned vs retained."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Total revenue by churn status
    rev = df.groupby("Churn")["TotalCharges"].sum()
    colors = [ACCENT_2, ACCENT_3]
    bars = axes[0].bar(rev.index, rev.values, color=colors, width=0.5,
                       edgecolor=DARK_BG, linewidth=1.5)
    for bar, val in zip(bars, rev.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
                     f"${val:,.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[0].set_title("Total Revenue by Status", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Total Revenue ($)")

    # Monthly charges distribution by churn
    for status, color in zip(["No", "Yes"], [ACCENT_2, ACCENT_3]):
        subset = df[df["Churn"] == status]["MonthlyCharges"]
        axes[1].hist(subset, bins=30, alpha=0.6, label=f"Churn={status}",
                     color=color, edgecolor=DARK_BG)
    axes[1].set_title("Monthly Charges Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Monthly Charges ($)")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.suptitle("Revenue Analysis", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "02_revenue_analysis.png")


def plot_demographics(df: pd.DataFrame):
    """Demographics vs churn — gender, senior, partner, dependents."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    demo_cols = ["gender", "SeniorCitizen", "Partner", "Dependents"]
    titles = ["Gender", "Senior Citizen", "Partner", "Dependents"]

    for ax, col, title in zip(axes.flatten(), demo_cols, titles):
        ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
        ct.plot(kind="bar", ax=ax, color=[ACCENT_2, ACCENT_3],
                edgecolor=DARK_BG, linewidth=1, rot=0)
        ax.set_title(f"{title} vs Churn", fontsize=13, fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("")
        ax.legend(title="Churn", loc="upper right")
        ax.set_ylim(0, 100)

    fig.suptitle("Demographics & Churn", fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "03_demographics_churn.png")


def plot_contract_analysis(df: pd.DataFrame):
    """Contract type vs churn — stacked bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ct = pd.crosstab(df["Contract"], df["Churn"])
    ct.plot(kind="bar", stacked=True, ax=ax, color=[ACCENT_2, ACCENT_3],
            edgecolor=DARK_BG, linewidth=1, rot=0)
    ax.set_title("Contract Type vs Churn", fontsize=16, fontweight="bold")
    ax.set_ylabel("Customer Count")
    ax.set_xlabel("Contract Type")
    ax.legend(title="Churn", loc="upper right")

    fig.tight_layout()
    _save(fig, "04_contract_churn.png")


def plot_tenure_analysis(df: pd.DataFrame):
    """Tenure vs churn — survival-style line chart + box plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # KDE of tenure by churn
    for status, color in zip(["No", "Yes"], [ACCENT_2, ACCENT_3]):
        subset = df[df["Churn"] == status]["tenure"]
        axes[0].hist(subset, bins=36, alpha=0.5, label=f"Churn={status}",
                     color=color, edgecolor=DARK_BG, density=True)
    axes[0].set_title("Tenure Distribution by Churn", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Tenure (months)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Box plot
    churn_data = [df[df["Churn"] == "No"]["tenure"], df[df["Churn"] == "Yes"]["tenure"]]
    bp = axes[1].boxplot(churn_data, labels=["Retained", "Churned"], patch_artist=True)
    for patch, color in zip(bp["boxes"], [ACCENT_2, ACCENT_3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color(TEXT_COLOR)
    axes[1].set_title("Tenure by Churn Status", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Tenure (months)")

    fig.suptitle("Tenure Analysis", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "05_tenure_analysis.png")


def plot_payment_method(df: pd.DataFrame):
    """Payment method vs churn — grouped bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ct = pd.crosstab(df["PaymentMethod"], df["Churn"], normalize="index") * 100
    ct.plot(kind="barh", ax=ax, color=[ACCENT_2, ACCENT_3],
            edgecolor=DARK_BG, linewidth=1)
    ax.set_title("Payment Method vs Churn Rate (%)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Percentage (%)")
    ax.set_ylabel("")
    ax.legend(title="Churn")

    fig.tight_layout()
    _save(fig, "06_payment_method_churn.png")


def plot_services_heatmap(df: pd.DataFrame):
    """Service usage heatmap — correlation with churn."""
    fig, ax = plt.subplots(figsize=(10, 8))

    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]

    # Create a binary version for correlation
    svc_df = pd.DataFrame()
    for col in service_cols:
        svc_df[col] = (df[col] == "Yes").astype(int) if col != "InternetService" else (df[col] != "No").astype(int)
    svc_df["Churn"] = (df["Churn"] == "Yes").astype(int)

    corr = svc_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, linewidths=0.5, linecolor=DARK_BG,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 10},
    )
    ax.set_title("Services & Churn Correlation Heatmap", fontsize=16, fontweight="bold", pad=15)

    fig.tight_layout()
    _save(fig, "07_services_heatmap.png")


def plot_correlation_matrix(df: pd.DataFrame):
    """Correlation matrix of numerical features."""
    fig, ax = plt.subplots(figsize=(10, 8))

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_Binary"]
    if "Churn_Binary" not in df.columns:
        df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
        center=0, ax=ax, linewidths=0.5, linecolor=DARK_BG,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 12},
    )
    ax.set_title("Numerical Features Correlation", fontsize=16, fontweight="bold", pad=15)

    fig.tight_layout()
    _save(fig, "08_correlation_matrix.png")


def run_eda(data_path: str = CLEAN_PATH) -> None:
    """Execute full EDA pipeline."""
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = load_data(data_path)
    print(f"\n📥 Loaded data: {df.shape}")

    # Ensure Churn_Binary exists
    if "Churn_Binary" not in df.columns:
        df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    print("\n📊 Generating visualizations...\n")

    plot_churn_distribution(df)
    plot_revenue_analysis(df)
    plot_demographics(df)
    plot_contract_analysis(df)
    plot_tenure_analysis(df)
    plot_payment_method(df)
    plot_services_heatmap(df)
    plot_correlation_matrix(df)

    print(f"\n✅ All EDA plots saved to: {os.path.abspath(REPORTS_DIR)}")

    # Print summary statistics
    print("\n── Summary Statistics ──")
    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"  Churn Rate:         {churn_rate:.1%}")
    print(f"  Total Customers:    {len(df):,}")
    print(f"  Avg Monthly Charge: ${df['MonthlyCharges'].mean():.2f}")
    print(f"  Avg Tenure:         {df['tenure'].mean():.1f} months")
    print(f"  Revenue at Risk:    ${df[df['Churn'] == 'Yes']['TotalCharges'].sum():,.0f}")


if __name__ == "__main__":
    run_eda()
