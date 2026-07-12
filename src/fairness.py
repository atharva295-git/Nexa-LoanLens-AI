# =============================================================
#  src/fairness.py  —  Bias Detection & Mitigation
#  Runs THIRD. Measures gender bias, applies Reweighing,
#  retrains a fair model, and plots before/after comparison.
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no popup window needed
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.preprocess import load_and_preprocess


def compute_fairness_metrics(y_true, y_pred, sensitive):
    """
    Computes fairness metrics manually using plain numpy/pandas.
    This avoids AIF360's NaN issues with scaled data.

    Disparate Impact (DI):
        = approval rate of females / approval rate of males
        Ideal = 1.0  |  Below 0.8 = legally biased (80% rule)

    Statistical Parity Difference (SPD):
        = approval rate of females - approval rate of males
        Ideal = 0.0
    """
    # Convert to numpy arrays for easy indexing
    y      = np.array(y_true)
    preds  = np.array(y_pred)
    sens   = np.array(sensitive)

    # Approval rates for each gender group
    male_mask   = sens == 1   # male   encoded as 1
    female_mask = sens == 0   # female encoded as 0

    male_approval   = preds[male_mask].mean()   if male_mask.sum()   > 0 else 0
    female_approval = preds[female_mask].mean() if female_mask.sum() > 0 else 0

    # Disparate Impact = female rate / male rate
    di  = female_approval / male_approval if male_approval > 0 else 0.0
    # Statistical Parity Difference = female rate - male rate
    spd = female_approval - male_approval

    return di, spd, male_approval, female_approval


def compute_reweighing_weights(X_train, y_train, sensitive_train):
    """
    Manually computes Reweighing sample weights.

    How Reweighing works:
    For each combination of (group, label), we compute:
        expected weight = P(group) * P(label)
        observed weight = P(group AND label)
        sample weight   = expected / observed

    Samples from underrepresented (group, label) pairs
    get weights > 1.0 so the model pays more attention to them.
    Samples from overrepresented pairs get weights < 1.0.
    """
    sens = np.array(sensitive_train)
    y    = np.array(y_train)
    n    = len(y)

    weights = np.ones(n)   # start with all weights = 1.0

    for group in [0, 1]:       # 0=female, 1=male
        for label in [0, 1]:   # 0=rejected, 1=approved
            # Mask for this (group, label) combination
            mask = (sens == group) & (y == label)

            p_group = (sens == group).mean()   # P(group)
            p_label = (y == label).mean()       # P(label)
            p_both  = mask.mean()               # P(group AND label)

            if p_both > 0:
                # Weight = what we expected / what we observed
                w = (p_group * p_label) / p_both
                weights[mask] = w

    print(f"  Weight range: {weights.min():.4f} — {weights.max():.4f}")
    print(f"  (>1.0 = underrepresented group gets more attention)")
    return weights


def run_fairness_pipeline():
    """
    Full fairness workflow:
    1. Load data
    2. Measure bias BEFORE mitigation (on training data labels)
    3. Compute Reweighing sample weights
    4. Retrain fair model with those weights
    5. Measure bias AFTER mitigation (on test set predictions)
    6. Plot comparison charts
    7. Save fair model
    """

    print("=" * 55)
    print("  FAIRNESS ANALYSIS")
    print("=" * 55)

    # Load preprocessed data
    (X_train, X_test,
     y_train, y_test,
     s_train, s_test,
     X_train_raw, X_test_raw) = load_and_preprocess()

    # ----------------------------------------------------------
    # BEFORE MITIGATION
    # Measure bias using the raw training labels (ground truth).
    # This tells us how biased the historical data is.
    # ----------------------------------------------------------
    print("\nMeasuring bias BEFORE mitigation (on training labels)...")

    # Use training labels as "predictions" to measure data-level bias
    di_before, spd_before, male_before, female_before = compute_fairness_metrics(
        y_train, y_train, s_train   # comparing labels to themselves = data bias
    )

    print(f"\n  [Before Mitigation — Data Level]")
    print(f"    Male approval rate   : {male_before:.1%}")
    print(f"    Female approval rate : {female_before:.1%}")
    print(f"    Gap                  : {abs(male_before - female_before):.1%}")
    print(f"    Disparate Impact     : {di_before:.4f}  (ideal = 1.0)")
    print(f"    Stat. Parity Diff    : {spd_before:.4f} (ideal = 0.0)")
    print(f"    Bias status          : {'BIASED (DI < 0.8)' if di_before < 0.8 else 'ACCEPTABLE'}")

    # ----------------------------------------------------------
    # APPLY REWEIGHING
    # Compute a weight for every training sample.
    # These weights tell the model which samples to focus on more.
    # ----------------------------------------------------------
    print("\nApplying Reweighing mitigation...")
    weights = compute_reweighing_weights(X_train, y_train, s_train)

    # Retrain Logistic Regression WITH the fairness weights
    lr_fair = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"   # also helps with class imbalance
    )
    lr_fair.fit(X_train, y_train, sample_weight=weights)
    print("  Fair model trained successfully.")

    # ----------------------------------------------------------
    # AFTER MITIGATION
    # Measure bias on the TEST SET predictions from the fair model.
    # Test set was never seen during training — honest evaluation.
    # ----------------------------------------------------------
    print("\nMeasuring bias AFTER mitigation (on test predictions)...")
    y_pred_fair = lr_fair.predict(X_test)

    di_after, spd_after, male_after, female_after = compute_fairness_metrics(
        y_test, y_pred_fair, s_test
    )

    acc_fair = accuracy_score(y_test, y_pred_fair)
    f1_fair  = f1_score(y_test, y_pred_fair)

    print(f"\n  [After Mitigation — Test Set Predictions]")
    print(f"    Male approval rate   : {male_after:.1%}")
    print(f"    Female approval rate : {female_after:.1%}")
    print(f"    Gap                  : {abs(male_after - female_after):.1%}")
    print(f"    Disparate Impact     : {di_after:.4f}  (ideal = 1.0)")
    print(f"    Stat. Parity Diff    : {spd_after:.4f} (ideal = 0.0)")
    print(f"    Bias status          : {'BIASED (DI < 0.8)' if di_after < 0.8 else 'ACCEPTABLE'}")
    print(f"\n  Fair model accuracy  : {acc_fair:.4f} ({acc_fair:.1%})")
    print(f"  Fair model F1 score  : {f1_fair:.4f}")

    # ----------------------------------------------------------
    # SUMMARY: show the improvement clearly
    # ----------------------------------------------------------
    print(f"\n  {'='*45}")
    print(f"  FAIRNESS IMPROVEMENT SUMMARY")
    print(f"  {'='*45}")
    print(f"  {'Metric':<30} {'Before':>8} {'After':>8}")
    print(f"  {'-'*46}")
    print(f"  {'Disparate Impact':<30} {di_before:>8.4f} {di_after:>8.4f}")
    print(f"  {'Stat. Parity Diff':<30} {spd_before:>8.4f} {spd_after:>8.4f}")
    print(f"  {'Male Approval Rate':<30} {male_before:>8.1%} {male_after:>8.1%}")
    print(f"  {'Female Approval Rate':<30} {female_before:>8.1%} {female_after:>8.1%}")
    print(f"  {'='*45}")

    # Store all results for the Streamlit app
    fairness_results = {
        "di_before":             di_before,
        "di_after":              di_after,
        "spd_before":            spd_before,
        "spd_after":             spd_after,
        "male_before":           male_before,
        "female_before":         female_before,
        "male_approval_after":   male_after,
        "female_approval_after": female_after,
        "accuracy_fair":         acc_fair,
        "f1_fair":               f1_fair,
    }

    # Generate the comparison charts
    _plot_fairness(fairness_results)

    # Save fair model and results to disk
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr_fair,          "models/lr_fair.pkl")
    joblib.dump(weights,          "models/sample_weights.pkl")
    joblib.dump(fairness_results, "models/fairness_results.pkl")
    print("\n  Saved: models/lr_fair.pkl")
    print("  Saved: models/fairness_results.pkl")
    print("\nFairness pipeline complete!")

    return lr_fair, fairness_results


def _plot_fairness(res):
    """
    Creates a 3-panel chart showing fairness metrics before vs after.
    Saved to models/fairness_comparison.png
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "Fairness Analysis: Before vs After Bias Mitigation (Reweighing)",
        fontsize=14, fontweight="bold", y=1.02
    )

    RED   = "#E85B5B"
    GREEN = "#2ECC8A"

    # --- Panel 1: Disparate Impact ---
    ax1 = axes[0]
    di_vals = [res["di_before"], res["di_after"]]
    # Clamp values to valid range for display
    di_vals = [max(0, min(v, 2.0)) for v in di_vals]

    bars = ax1.bar(
        ["Before\nMitigation", "After\nMitigation"],
        di_vals, color=[RED, GREEN], width=0.45,
        edgecolor="white", linewidth=1.5
    )
    ax1.axhline(y=0.8, color="black", linestyle="--",
                linewidth=1.5, label="80% rule (0.80)")
    ax1.axhline(y=1.0, color="gray",  linestyle=":",
                linewidth=1.0, alpha=0.6, label="Perfect (1.00)")
    for bar, val in zip(bars, di_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center",
                 fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 1.3)
    ax1.set_ylabel("Disparate Impact Score")
    ax1.set_title("Disparate Impact\n(higher = fairer)", fontweight="bold")
    ax1.legend(fontsize=8)

    # --- Panel 2: Approval rates by gender ---
    ax2 = axes[1]
    x = np.arange(2)   # positions for Male, Female
    w = 0.3

    before_vals = [res["male_before"],         res["female_before"]]
    after_vals  = [res["male_approval_after"],  res["female_approval_after"]]

    b1 = ax2.bar(x - w/2, before_vals, w,
                 label="Before", color=RED,   alpha=0.85, edgecolor="white")
    b2 = ax2.bar(x + w/2, after_vals,  w,
                 label="After",  color=GREEN, alpha=0.85, edgecolor="white")

    for bars in [b1, b2]:
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{bar.get_height():.1%}",
                     ha="center", fontsize=10, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Male", "Female"])
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Approval Rate")
    ax2.set_title("Approval Rate by Gender\n(closer gap = fairer)", fontweight="bold")
    ax2.legend(fontsize=9)

    # --- Panel 3: Statistical Parity Difference ---
    ax3 = axes[2]
    spd_vals = [abs(res["spd_before"]), abs(res["spd_after"])]

    # Guard against NaN or zero — set safe ylim
    max_spd = max(spd_vals) if max(spd_vals) > 0 else 0.1

    bars3 = ax3.bar(
        ["Before\nMitigation", "After\nMitigation"],
        spd_vals, color=[RED, GREEN],
        width=0.45, edgecolor="white", linewidth=1.5
    )
    for bar, val in zip(bars3, spd_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max_spd * 0.03,
                 f"{val:.3f}", ha="center",
                 fontsize=13, fontweight="bold")
    ax3.set_ylim(0, max_spd * 1.5)
    ax3.set_ylabel("|Statistical Parity Difference|")
    ax3.set_title("Stat. Parity Difference\n(lower = fairer)", fontweight="bold")

    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/fairness_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: models/fairness_comparison.png")


# Run directly:
#   python -m src.fairness
if __name__ == "__main__":
    run_fairness_pipeline()