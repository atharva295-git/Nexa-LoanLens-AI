# =============================================================
#  src/model.py  —  Model Training & Evaluation
#  Runs SECOND. Trains two ML models, evaluates them, saves
#  them to disk, and generates comparison charts.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve
)

# Import our own preprocessing function from preprocess.py
from src.preprocess import load_and_preprocess


def train_models():
    """
    Loads preprocessed data, trains Logistic Regression and XGBoost,
    evaluates both, saves model files, and produces evaluation charts.
    """

    # ----------------------------------------------------------
    # Load the preprocessed data from preprocess.py
    # We get back 8 objects — scaled features, raw features,
    # labels, and the sensitive attribute (Sex column)
    # ----------------------------------------------------------
    print("=" * 55)
    print("  MODEL TRAINING")
    print("=" * 55)

    (X_train, X_test,
     y_train, y_test,
     s_train, s_test,
     X_train_raw, X_test_raw) = load_and_preprocess()

    # ----------------------------------------------------------
    # Define the two models we'll train and compare.
    # Logistic Regression: simple, fast, interpretable.
    # XGBoost: more powerful tree-based model, usually more accurate.
    # We compare both so the report can show the trade-off.
    # ----------------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,     # max training iterations before giving up
            random_state=42,   # for reproducibility
            class_weight="balanced"  # handles imbalanced good/bad ratio
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,       # number of decision trees to build
            max_depth=4,            # how deep each tree can grow
            learning_rate=0.05,     # how fast the model learns (slow = better)
            random_state=42,
            eval_metric="logloss",  # loss function for binary classification
            verbosity=0             # suppress XGBoost's own print output
        ),
    }

    results = {}  # store metrics for each model

    for name, model in models.items():
        print(f"\n  Training: {name}...")

        # Train the model on the scaled training data
        model.fit(X_train, y_train)

        # Use the trained model to predict on the test set
        y_pred = model.predict(X_test)

        # predict_proba gives probabilities (0.0 to 1.0) instead of just 0/1
        # Used for AUC-ROC which measures ranking ability
        y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1

        # Calculate all evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "model":  model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "accuracy": acc,
            "f1":       f1,
            "auc":      auc,
        }

        print(f"    Accuracy  : {acc:.4f}  ({acc:.1%})")
        print(f"    F1 Score  : {f1:.4f}")
        print(f"    AUC-ROC   : {auc:.4f}")
        print(f"\n  Classification Report ({name}):")
        print(classification_report(y_test, y_pred,
              target_names=["Rejected (0)", "Approved (1)"]))

        # Save model to disk so other scripts and app.py can load it
        safe_name = name.replace(" ", "_")
        joblib.dump(model, f"models/{safe_name}.pkl")
        print(f"  Saved: models/{safe_name}.pkl")

    # ----------------------------------------------------------
    # Generate evaluation charts
    # Three charts side by side:
    #   1. Confusion matrices for both models
    #   2. ROC curves for both models
    #   3. Feature importance (coefficients from Logistic Regression)
    # ----------------------------------------------------------
    print("\n  Generating evaluation charts...")
    _plot_evaluation(results, X_train, y_test)

    return results


def _plot_evaluation(results, X_train, y_test):
    """
    Creates a 2x2 chart grid saved as models/model_evaluation.png
    """
    # Set a clean, modern style for all charts
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Model Evaluation Report", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    colors = {"Logistic Regression": "#5B6FE8", "XGBoost": "#E85B5B"}

    # --- Chart 1 & 2: Confusion Matrices (one per model) ---
    for i, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, i])
        cm = confusion_matrix(y_test, res["y_pred"])

        # Normalize confusion matrix to show percentages
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=True, fmt=".1%", cmap="Blues",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"],
            ax=ax, linewidths=0.5, cbar=False,
            annot_kws={"size": 12}
        )
        # Also show raw counts in each cell
        for j in range(2):
            for k in range(2):
                ax.text(k + 0.5, j + 0.72, f"n={cm[j,k]}",
                        ha="center", va="center", fontsize=9,
                        color="gray")
        ax.set_title(f"{name}\nConfusion Matrix", fontweight="bold", pad=10)
        ax.set_ylabel("Actual Label")
        ax.set_xlabel("Predicted Label")

    # --- Chart 3: ROC Curves (both models on same chart) ---
    ax3 = fig.add_subplot(gs[1, 0])
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax3.plot(fpr, tpr,
                 label=f"{name}  (AUC = {res['auc']:.3f})",
                 color=colors[name], linewidth=2)
    # Diagonal line = random classifier baseline
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random classifier")
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curves", fontweight="bold")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1])

    # --- Chart 4: Metric comparison bar chart ---
    ax4 = fig.add_subplot(gs[1, 1])
    metric_names = ["Accuracy", "F1 Score", "AUC-ROC"]
    x = np.arange(len(metric_names))
    width = 0.3

    for i, (name, res) in enumerate(results.items()):
        vals = [res["accuracy"], res["f1"], res["auc"]]
        bars = ax4.bar(x + i * width, vals, width,
                       label=name, color=colors[name], alpha=0.85)
        # Add value labels on top of bars
        for bar, val in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax4.set_xticks(x + width / 2)
    ax4.set_xticklabels(metric_names)
    ax4.set_ylim([0, 1.12])
    ax4.set_ylabel("Score")
    ax4.set_title("Model Metrics Comparison", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.axhline(y=0.8, color="gray", linestyle="--", alpha=0.4, linewidth=1)

    plt.savefig("models/model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: models/model_evaluation.png")


# Run this file directly to train models:
#   python src/model.py
if __name__ == "__main__":
    train_models()
