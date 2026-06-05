"""
Explainable AI Module
=====================
SHAP-based model explanations for global and local interpretability.
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
import joblib
import shap

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features_churn.csv")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "cleaned_churn.csv")

# Style
DARK_BG = "#0E1117"
CARD_BG = "#1A1A2E"
TEXT_COLOR = "#E0E0E0"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#333355", "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR, "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR, "font.size": 11,
})


def load_artifacts():
    """Load model, scaler, and feature names."""
    models_dir = os.path.abspath(MODELS_DIR)
    model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_names.json")) as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


def load_test_data(feature_names: list) -> tuple[np.ndarray, pd.DataFrame]:
    """Load feature-engineered data for SHAP analysis."""
    features_path = os.path.abspath(FEATURES_PATH)
    df = pd.read_csv(features_path)

    X = df[feature_names].values
    return X, df


def compute_shap_values(model, X: np.ndarray, scaler, feature_names: list):
    """Compute SHAP values using the appropriate explainer."""
    X_scaled = scaler.transform(X)

    model_type = type(model).__name__

    if model_type in ("RandomForestClassifier", "XGBClassifier"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        # For binary classification, TreeExplainer may return a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (churn)
    else:
        # Logistic Regression → LinearExplainer
        explainer = shap.LinearExplainer(model, X_scaled)
        shap_values = explainer.shap_values(X_scaled)

    return shap_values, explainer


def plot_feature_importance(shap_values: np.ndarray, feature_names: list, top_n: int = 15):
    """Global feature importance bar chart."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))

    y_pos = range(len(top_idx))
    ax.barh(y_pos, mean_abs_shap[top_idx], color=colors,
            edgecolor=DARK_BG, linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Top Feature Importance (SHAP)", fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.abspath(REPORTS_DIR), "14_feature_importance.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  📊 Saved: 14_feature_importance.png")


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray, feature_names: list):
    """SHAP summary beeswarm plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, feature_names=feature_names,
        show=False, max_display=15, plot_size=None,
    )
    plt.title("SHAP Summary (Beeswarm)", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()

    out = os.path.join(os.path.abspath(REPORTS_DIR), "15_shap_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  📊 Saved: 15_shap_summary.png")


def plot_dependence(shap_values: np.ndarray, X: np.ndarray, feature_names: list, top_n: int = 3):
    """SHAP dependence plots for top features."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-top_n:][::-1]

    fig, axes = plt.subplots(1, top_n, figsize=(6 * top_n, 5))
    if top_n == 1:
        axes = [axes]

    for ax, idx in zip(axes, top_idx):
        shap.dependence_plot(
            idx, shap_values, X, feature_names=feature_names,
            ax=ax, show=False,
        )
        ax.set_title(f"{feature_names[idx]}", fontsize=13, fontweight="bold")

    fig.suptitle("SHAP Dependence Plots", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    out = os.path.join(os.path.abspath(REPORTS_DIR), "16_shap_dependence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  📊 Saved: 16_shap_dependence.png")


def generate_insights(shap_values: np.ndarray, X: np.ndarray,
                      feature_names: list, df: pd.DataFrame) -> list[dict]:
    """Generate natural-language business insights from SHAP analysis."""
    insights = []

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = [feature_names[i] for i in np.argsort(mean_abs_shap)[-5:][::-1]]

    # Load original cleaned data for readable insights
    clean_path = os.path.abspath(CLEAN_PATH)
    if os.path.exists(clean_path):
        df_clean = pd.read_csv(clean_path)
        if "Churn_Binary" not in df_clean.columns:
            df_clean["Churn_Binary"] = (df_clean["Churn"] == "Yes").astype(int)

        # Contract analysis
        if "Contract" in df_clean.columns:
            mtm_churn = df_clean[df_clean["Contract"] == "Month-to-month"]["Churn_Binary"].mean()
            other_churn = df_clean[df_clean["Contract"] != "Month-to-month"]["Churn_Binary"].mean()
            if other_churn > 0:
                ratio = mtm_churn / other_churn
                insights.append({
                    "category": "Contract",
                    "insight": f"Customers on month-to-month contracts are {ratio:.1f}x more likely to churn.",
                    "churn_rate_mtm": round(mtm_churn, 3),
                    "churn_rate_other": round(other_churn, 3),
                    "priority": "high",
                })

        # Tenure + charges analysis
        short_tenure_high_charge = df_clean[
            (df_clean["tenure"] < 12) & (df_clean["MonthlyCharges"] > 70)
        ]
        if len(short_tenure_high_charge) > 0:
            cr = short_tenure_high_charge["Churn_Binary"].mean()
            insights.append({
                "category": "Tenure & Charges",
                "insight": f"Customers with tenure < 12 months and monthly charges > $70 "
                           f"have a {cr:.0%} churn rate.",
                "affected_customers": len(short_tenure_high_charge),
                "churn_rate": round(cr, 3),
                "priority": "high",
            })

        # Internet service analysis
        if "InternetService" in df_clean.columns:
            fiber_churn = df_clean[df_clean["InternetService"] == "Fiber optic"]["Churn_Binary"].mean()
            dsl_churn = df_clean[df_clean["InternetService"] == "DSL"]["Churn_Binary"].mean()
            insights.append({
                "category": "Internet Service",
                "insight": f"Fiber optic customers churn at {fiber_churn:.0%} vs "
                           f"{dsl_churn:.0%} for DSL — investigate service quality.",
                "fiber_churn_rate": round(fiber_churn, 3),
                "dsl_churn_rate": round(dsl_churn, 3),
                "priority": "medium",
            })

        # Senior citizen analysis
        sr_churn = df_clean[df_clean["SeniorCitizen"] == 1]["Churn_Binary"].mean()
        non_sr_churn = df_clean[df_clean["SeniorCitizen"] == 0]["Churn_Binary"].mean()
        insights.append({
            "category": "Demographics",
            "insight": f"Senior citizens have a {sr_churn:.0%} churn rate vs "
                       f"{non_sr_churn:.0%} for non-seniors.",
            "priority": "medium",
        })

        # Payment method
        if "PaymentMethod" in df_clean.columns:
            ec_churn = df_clean[df_clean["PaymentMethod"] == "Electronic check"]["Churn_Binary"].mean()
            auto_churn = df_clean[
                df_clean["PaymentMethod"].str.contains("automatic", case=False)
            ]["Churn_Binary"].mean()
            insights.append({
                "category": "Payment",
                "insight": f"Electronic check users churn at {ec_churn:.0%} vs "
                           f"{auto_churn:.0%} for auto-payment customers.",
                "priority": "medium",
            })

    insights.append({
        "category": "Model",
        "insight": f"Top churn drivers: {', '.join(top_features[:3])}.",
        "top_features": top_features,
        "priority": "info",
    })

    return insights


def run_explainability():
    """Execute the full explainability pipeline."""
    print("=" * 60)
    print("  EXPLAINABLE AI — SHAP ANALYSIS")
    print("=" * 60)

    model, scaler, feature_names = load_artifacts()
    print(f"\n📥 Loaded model: {type(model).__name__}")
    print(f"   Features: {len(feature_names)}")

    X, df = load_test_data(feature_names)
    print(f"   Data shape: {X.shape}")

    # Compute SHAP values (use sample for speed)
    sample_size = min(1000, len(X))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X), size=sample_size, replace=False)
    X_sample = X[sample_idx]

    print(f"\n🔍 Computing SHAP values (sample of {sample_size})...")
    shap_values, explainer = compute_shap_values(model, X_sample, scaler, feature_names)
    print(f"   SHAP values shape: {shap_values.shape}")

    # Generate plots
    print("\n📊 Generating SHAP visualizations...")
    plot_feature_importance(shap_values, feature_names)
    plot_shap_summary(shap_values, X_sample, feature_names)
    plot_dependence(shap_values, X_sample, feature_names)

    # Generate insights
    print("\n💡 Generating business insights...")
    insights = generate_insights(shap_values, X_sample, feature_names, df)

    insights_path = os.path.join(os.path.abspath(REPORTS_DIR), "insights.json")
    with open(insights_path, "w") as f:
        json.dump(insights, f, indent=2)
    print(f"   Saved to: {insights_path}")

    print("\n── Key Insights ──")
    for ins in insights:
        priority_icon = {"high": "🔴", "medium": "🟡", "info": "🔵"}.get(ins["priority"], "⚪")
        print(f"  {priority_icon} [{ins['category']}] {ins['insight']}")

    print(f"\n✅ Explainability analysis complete!")


if __name__ == "__main__":
    run_explainability()
