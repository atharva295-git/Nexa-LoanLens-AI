# =============================================================
#  src/explainer.py  —  Counterfactual Explanations
#  Uses a custom counterfactual search instead of DiCE,
#  which has compatibility issues with pandas 3.0+.
#
#  How it works:
#  For each rejected applicant, we systematically try small
#  changes to their features and find the minimal combination
#  that flips the model's decision from REJECTED → APPROVED.
# =============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import joblib
from sklearn.pipeline import Pipeline
from itertools import product

from src.preprocess import load_and_preprocess


def build_pipeline(X_train_raw, y_train):
    """
    Builds a sklearn Pipeline: scaler → fair model.
    This lets us predict on raw (unscaled) data directly.
    """
    scaler  = joblib.load("models/scaler.pkl")
    lr_fair = joblib.load("models/lr_fair.pkl")
    pipe = Pipeline([("scaler", scaler), ("clf", lr_fair)])
    pipe.fit(X_train_raw, y_train)
    return pipe


def find_counterfactuals(applicant_row, pipe, X_train_raw, n_cfs=3):
    """
    Custom counterfactual search using feature perturbation.

    Strategy:
    1. For each continuous feature (Age, Credit amount, Duration),
       try a range of values above and below the original.
    2. For each categorical feature, try all possible valid values.
    3. Find combinations that flip the prediction to APPROVED.
    4. Return the top counterfactuals with fewest changes.

    Parameters
    ----------
    applicant_row : pd.Series — the rejected applicant's features
    pipe          : fitted Pipeline (scaler + model)
    X_train_raw   : training data to get valid value ranges
    n_cfs         : number of counterfactuals to return
    """

    # Define which features are continuous vs categorical
    continuous_cols   = ["Age", "Credit amount", "Duration"]
    categorical_cols  = ["Sex", "Job", "Housing",
                         "Saving accounts", "Checking account", "Purpose"]

    # Get the range of values seen in training data for continuous features
    # We'll try perturbations within this range
    cont_ranges = {}
    for col in continuous_cols:
        col_min = X_train_raw[col].min()
        col_max = X_train_raw[col].max()
        original = applicant_row[col]

        # Try 8 values: some lower, some higher than original
        step = (col_max - col_min) / 8
        candidates = sorted(set([
            max(col_min, original - step * 2),
            max(col_min, original - step),
            original,                           # keep original as option
            min(col_max, original + step),
            min(col_max, original + step * 2),
        ]))
        cont_ranges[col] = candidates

    # For categorical features, try all unique values seen in training
    cat_values = {}
    for col in categorical_cols:
        cat_values[col] = sorted(X_train_raw[col].unique().tolist())

    counterfactuals = []   # will store (num_changes, changed_features, new_row)

    # ----------------------------------------------------------
    # Search strategy: try changing ONE feature at a time first,
    # then TWO features, then THREE.
    # This finds the MINIMAL change needed.
    # ----------------------------------------------------------

    all_cols = continuous_cols + categorical_cols

    # Try changing 1, 2, or 3 features
    for n_changes in [1, 2, 3]:
        if len(counterfactuals) >= n_cfs:
            break

        # Pick combinations of n_changes features to modify
        from itertools import combinations
        for cols_to_change in combinations(all_cols, n_changes):

            if len(counterfactuals) >= n_cfs * 3:
                break

            # Build candidate value lists for each column being changed
            value_options = []
            for col in cols_to_change:
                if col in continuous_cols:
                    # Try all candidate values except the original
                    opts = [v for v in cont_ranges[col]
                            if abs(v - applicant_row[col]) > 0.01]
                    value_options.append(opts if opts else [applicant_row[col]])
                else:
                    # Try all values except the original
                    opts = [v for v in cat_values[col]
                            if v != applicant_row[col]]
                    value_options.append(opts if opts else [applicant_row[col]])

            # Try every combination of candidate values
            for combo in product(*value_options):
                # Build the modified applicant row
                new_row = applicant_row.astype(float).copy()
                changes = {}
                for col, new_val in zip(cols_to_change, combo):
                    if abs(new_val - applicant_row[col]) > 0.01:
                        new_row[col] = new_val
                        changes[col] = (applicant_row[col], new_val)

                if not changes:
                    continue

                # Check if this change flips the prediction
                new_df   = new_row.to_frame().T.reset_index(drop=True)
                new_pred = pipe.predict(new_df)[0]
                new_prob = pipe.predict_proba(new_df)[0][1]

                if new_pred == 1:   # APPROVED!
                    counterfactuals.append({
                        "n_changes": len(changes),
                        "changes":   changes,
                        "new_row":   new_row,
                        "prob":      new_prob,
                    })

        if counterfactuals:
            break   # found some — don't need to try more changes

    # Sort by fewest changes, then highest approval probability
    counterfactuals.sort(key=lambda x: (x["n_changes"], -x["prob"]))

    return counterfactuals[:n_cfs]


def format_explanation(changes, reverse_enc):
    """
    Converts a dict of {feature: (original, new)} changes
    into human-readable text.

    Example output:
      "Decrease 'Duration' from 48 to 24 months"
      "Improve 'Saving accounts' from unknown to little"
    """
    lines = []
    for col, (orig, new) in changes.items():
        # Decode categorical values back to text
        if col in reverse_enc:
            orig_disp = reverse_enc[col].get(int(round(orig)), str(orig))
            new_disp  = reverse_enc[col].get(int(round(new)),  str(new))
        else:
            orig_disp = f"{orig:.0f}"
            new_disp  = f"{new:.0f}"

        direction = "Increase" if new > orig else "Decrease"
        lines.append(f"  {direction} '{col}': {orig_disp} → {new_disp}")
    return lines


def generate_explanations(n_applicants=3, n_cfs=3):
    """
    Main function: finds rejected applicants and generates
    counterfactual explanations for each one.
    """

    print("=" * 55)
    print("  COUNTERFACTUAL EXPLANATIONS")
    print("=" * 55)

    # Load data
    (X_train_sc, X_test_sc,
     y_train, y_test,
     s_train, s_test,
     X_train_raw, X_test_raw) = load_and_preprocess()

    # Build pipeline
    pipe = build_pipeline(X_train_raw, y_train)
    print("\n  Pipeline ready.")

    # Load encoders for human-readable output
    encoders    = joblib.load("models/encoders.pkl")
    reverse_enc = {col: {i: lbl for i, lbl in enumerate(le.classes_)}
                   for col, le in encoders.items()}

    # Find rejected applicants
    test_preds = pipe.predict(X_test_raw)
    rejected   = X_test_raw[test_preds == 0].head(n_applicants)

    print(f"  Found {(test_preds==0).sum()} rejected applicants in test set.")
    print(f"  Generating explanations for {len(rejected)}...\n")

    all_explanations = []

    for i, (idx, row) in enumerate(rejected.iterrows()):
        print(f"  --- Applicant {i+1} (index: {idx}) ---")
        print(f"  Profile:")
        for col in X_test_raw.columns:
            val     = row[col]
            display = reverse_enc[col].get(int(val), val) if col in reverse_enc else val
            print(f"    {col:22s}: {display}")

        print(f"\n  Searching for minimal changes to get APPROVED...")

        # Run our custom counterfactual search
        cfs = find_counterfactuals(row, pipe, X_train_raw, n_cfs=n_cfs)

        if cfs:
            print(f"  Found {len(cfs)} counterfactual(s):\n")
            for j, cf in enumerate(cfs):
                print(f"  Option {j+1} "
                      f"(approval probability: {cf['prob']:.1%}):")
                lines = format_explanation(cf["changes"], reverse_enc)
                for line in lines:
                    print(line)
                print()
        else:
            print("  No counterfactual found with up to 3 changes.")
            print("  This applicant's profile is very far from approval.\n")

        all_explanations.append({
            "applicant_idx": idx,
            "profile":       row.to_dict(),
            "counterfactuals": cfs
        })

    # Save explanations to disk for the app
    joblib.dump(all_explanations, "models/explanations.pkl")
    print("  Saved: models/explanations.pkl")
    print("\nExplainer complete!")
    return all_explanations


def explain_single(applicant_dict, pipe=None):
    """
    Generates a counterfactual explanation for ONE applicant.
    Used by the Streamlit app when a user submits their details.

    Returns a list of plain-English suggestion strings.
    """
    (_, _, y_train, _, _, _, X_train_raw, _) = load_and_preprocess()

    if pipe is None:
        pipe = build_pipeline(X_train_raw, y_train)

    encoders    = joblib.load("models/encoders.pkl")
    reverse_enc = {col: {i: lbl for i, lbl in enumerate(le.classes_)}
                   for col, le in encoders.items()}

    row = pd.Series(applicant_dict)
    # Ask for more candidates than we need so we have room to skip
    # unusable ones (see below) and still return something real.
    cfs = find_counterfactuals(row, pipe, X_train_raw, n_cfs=8)

    if not cfs:
        return [
            "Reduce loan **Duration**",
            "Reduce **Credit amount**",
            "Improve **Saving accounts** status",
        ]

    # Values that make for confusing / non-actionable advice
    # (e.g. telling someone to change an account status "to unknown"
    # isn't meaningful to a real applicant, even though the model
    # may technically rate it as lower-risk due to dataset quirks).
    UNUSABLE = {"unknown", "nan", "none"}

    # Try each counterfactual (best first) and use the first one
    # whose suggested targets are all genuinely actionable. This
    # avoids collapsing to zero suggestions just because the single
    # best-scoring counterfactual happens to be unusable.
    for cf in cfs:
        suggestions = []
        usable = True
        for col, (orig, new) in cf["changes"].items():
            if col in reverse_enc:
                orig_disp = reverse_enc[col].get(int(round(orig)), str(orig))
                new_disp  = reverse_enc[col].get(int(round(new)),  str(new))
            else:
                orig_disp = f"{orig:.0f}"
                new_disp  = f"{new:.0f}"

            if str(new_disp).strip().lower() in UNUSABLE:
                usable = False
                break

            direction = "Increase" if new > orig else "Decrease"
            suggestions.append(f"{direction} **{col}** from {orig_disp} to {new_disp}")

        if usable and suggestions:
            return suggestions

    # Every counterfactual we found only worked by moving to an
    # unusable value (e.g. "unknown" account status) — signal this
    # explicitly instead of falling back to generic, possibly
    # contradictory advice.
    return []


# Run directly:
#   python -m src.explainer
if __name__ == "__main__":
    generate_explanations()