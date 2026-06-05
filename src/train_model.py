"""
Churn Prediction — Model Training Pipeline
============================================
Trains Logistic Regression, Random Forest, and XGBoost.
Compares models, selects the best, and saves artifacts.
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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed — will skip XGBoost model.")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("⚠️  imbalanced-learn not installed — skipping SMOTE.")

FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features_churn.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG = "#0E1117"
CARD_BG = "#1A1A2E"
TEXT_COLOR = "#E0E0E0"
COLORS = ["#6C63FF", "#00D4AA", "#FF6B6B"]

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#333355", "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR, "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR, "grid.color": "#333355", "grid.alpha": 0.3,
    "font.size": 11,
})


def load_features(path: str = FEATURES_PATH) -> pd.DataFrame:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print(f"❌ Feature data not found at {path}")
        sys.exit(1)
    return pd.read_csv(path)


def prepare_train_data(df: pd.DataFrame) -> tuple:
    """Prepare X, y and split into train/test."""
    # Target
    if "Churn_Binary" not in df.columns:
        df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)
    y = df["Churn_Binary"]

    # Drop non-feature columns
    drop_cols = [
        "customerID", "Churn", "Churn_Binary", "gender", "tenure_group",
        # Original categorical columns (we use encoded versions)
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    ]
    # Also drop any columns that are still object type
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    feature_names = X.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE for class imbalance
    if HAS_SMOTE:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"   SMOTE applied: {len(X_train)} training samples (balanced)")

    return X_train, X_test, y_train, y_test, scaler, feature_names


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    return metrics


def train_all_models(X_train, X_test, y_train, y_test) -> list[tuple]:
    """Train all models and return (name, model, metrics) tuples."""
    results = []

    # ── Logistic Regression ────────────────────────────────────────────────
    print("\n  🔄 Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_params = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}
    lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring="f1", n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    lr_best = lr_grid.best_estimator_
    lr_metrics = evaluate_model(lr_best, X_test, y_test, "Logistic Regression")
    results.append(("Logistic Regression", lr_best, lr_metrics))
    print(f"     F1: {lr_metrics['f1_score']:.4f} | AUC: {lr_metrics['roc_auc']:.4f}")

    # ── Random Forest ──────────────────────────────────────────────────────
    print("  🔄 Training Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring="f1", n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    rf_metrics = evaluate_model(rf_best, X_test, y_test, "Random Forest")
    results.append(("Random Forest", rf_best, rf_metrics))
    print(f"     F1: {rf_metrics['f1_score']:.4f} | AUC: {rf_metrics['roc_auc']:.4f}")

    # ── XGBoost ────────────────────────────────────────────────────────────
    if HAS_XGBOOST:
        print("  🔄 Training XGBoost...")
        xgb = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42,
            verbosity=0,
        )
        xgb_params = {
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6, 8],
            "n_estimators": [100, 200],
        }
        xgb_grid = GridSearchCV(xgb, xgb_params, cv=5, scoring="f1", n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        xgb_best = xgb_grid.best_estimator_
        xgb_metrics = evaluate_model(xgb_best, X_test, y_test, "XGBoost")
        results.append(("XGBoost", xgb_best, xgb_metrics))
        print(f"     F1: {xgb_metrics['f1_score']:.4f} | AUC: {xgb_metrics['roc_auc']:.4f}")

    return results


def plot_model_comparison(results: list[tuple]):
    """Bar chart comparing model metrics."""
    metrics_list = [r[2] for r in results]
    names = [r[0] for r in results]
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_keys))
    width = 0.25

    for i, (name, _, metrics) in enumerate(results):
        values = [metrics[k] for k in metric_keys]
        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=COLORS[i],
                      edgecolor=DARK_BG, linewidth=1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"])
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=16, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(os.path.abspath(REPORTS_DIR), "11_model_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"\n  📊 Saved: 11_model_comparison.png")


def plot_roc_curves(results: list[tuple], X_test, y_test):
    """ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for i, (name, model, metrics) in enumerate(results):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, color=COLORS[i], linewidth=2,
                label=f"{name} (AUC={metrics['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1)
    ax.set_title("ROC Curves", fontsize=16, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out = os.path.join(os.path.abspath(REPORTS_DIR), "12_roc_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  📊 Saved: 12_roc_curves.png")


def plot_confusion_matrices(results: list[tuple], X_test, y_test):
    """Confusion matrices for all models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model, _) in zip(axes, results):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap="Blues", alpha=0.8)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=16, fontweight="bold", color="white")
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Retained", "Churned"])
        ax.set_yticklabels(["Retained", "Churned"])

    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(os.path.abspath(REPORTS_DIR), "13_confusion_matrices.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  📊 Saved: 13_confusion_matrices.png")


def save_artifacts(best_name, best_model, scaler, feature_names, all_metrics):
    """Save model, scaler, and metadata."""
    models_dir = os.path.abspath(MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)

    # Save best model
    model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n  💾 Best model ({best_name}) saved to: {model_path}")

    # Save scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  💾 Scaler saved to: {scaler_path}")

    # Save feature names
    fn_path = os.path.join(models_dir, "feature_names.json")
    with open(fn_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"  💾 Feature names saved to: {fn_path}")

    # Save all metrics
    metrics_path = os.path.join(models_dir, "model_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  💾 Model comparison saved to: {metrics_path}")


def run_training(features_path: str = FEATURES_PATH) -> tuple:
    """Execute the full training pipeline."""
    print("=" * 60)
    print("  CHURN PREDICTION — MODEL TRAINING")
    print("=" * 60)

    df = load_features(features_path)
    print(f"\n📥 Loaded feature data: {df.shape}")

    # Prepare data
    print("\n🔧 Preparing training data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_data(df)
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

    # Train models
    print("\n🚀 Training models...")
    results = train_all_models(X_train, X_test, y_train, y_test)

    # Plots
    print("\n📊 Generating evaluation plots...")
    plot_model_comparison(results)
    plot_roc_curves(results, X_test, y_test)
    plot_confusion_matrices(results, X_test, y_test)

    # Select best model (by F1)
    best_idx = np.argmax([r[2]["f1_score"] for r in results])
    best_name, best_model, best_metrics = results[best_idx]

    print(f"\n🏆 Best model: {best_name}")
    print(f"   Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall:    {best_metrics['recall']:.4f}")
    print(f"   F1 Score:  {best_metrics['f1_score']:.4f}")
    print(f"   ROC-AUC:   {best_metrics['roc_auc']:.4f}")

    # Save
    all_metrics = [r[2] for r in results]
    save_artifacts(best_name, best_model, scaler, feature_names, all_metrics)

    return best_model, scaler, feature_names, results


if __name__ == "__main__":
    run_training()
