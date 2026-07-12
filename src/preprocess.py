# =============================================================
#  src/preprocess.py  —  Data Loading & Preprocessing
#  This is the FIRST file that runs. Every other file depends
#  on the data that this file prepares.
# =============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


def load_and_preprocess(path="data/german_credit_data.csv"):
    """
    Loads the raw German Credit CSV, cleans it, encodes categories,
    scales numeric columns, and splits into train/test sets.
    """

    # ----------------------------------------------------------
    # STEP 1: Load the CSV
    # This dataset has an unnamed index column — index_col=0 drops it.
    # Then we check whether 'Risk' is a column or the index,
    # and reset accordingly so we always have it as a regular column.
    # ----------------------------------------------------------
    print("Loading dataset...")
    df = pd.read_csv(path, index_col=0)

    # If Risk ended up as the index, move it back to a column
    if df.index.name == "Risk":
        df = df.reset_index()

    # If Risk is still not in columns, try reading without index_col
    if "Risk" not in df.columns:
        df = pd.read_csv(path)
        # Drop any unnamed columns (e.g. 'Unnamed: 0')
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    print(f"  Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # Confirm Risk column exists
    if "Risk" not in df.columns:
        raise ValueError(
            "Could not find 'Risk' column. "
            f"Available columns: {list(df.columns)}"
        )

    # ----------------------------------------------------------
    # STEP 2: Handle missing values
    # Use the new pandas syntax to avoid Copy-on-Write warnings
    # df[col] = df[col].fillna(value)  instead of inplace=True
    # ----------------------------------------------------------
    df["Saving accounts"]  = df["Saving accounts"].fillna("unknown")
    df["Checking account"] = df["Checking account"].fillna("unknown")

    # ----------------------------------------------------------
    # STEP 3: Encode categorical (text) columns into numbers
    # LabelEncoder converts text → integer
    # We save each encoder so app.py can encode user inputs later
    # ----------------------------------------------------------
    categorical_cols = [
        "Sex",               # protected attribute — used in fairness checks
        "Housing",
        "Saving accounts",
        "Checking account",
        "Purpose",
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"  Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ----------------------------------------------------------
    # STEP 4: Encode the target column
    # 'good' = 1 (approved), 'bad' = 0 (rejected)
    # ----------------------------------------------------------
    df["Risk"] = df["Risk"].map({"good": 1, "bad": 0})
    print(f"\n  Class distribution (0=rejected, 1=approved):")
    print(f"  {df['Risk'].value_counts().to_dict()}")

    # ----------------------------------------------------------
    # STEP 5: Separate features (X) from label (y)
    # sensitive = Sex column, used for fairness measurement
    # ----------------------------------------------------------
    X = df.drop("Risk", axis=1)
    y = df["Risk"]
    sensitive = X["Sex"]

    print(f"\n  Feature columns: {list(X.columns)}")

    # ----------------------------------------------------------
    # STEP 6: Train / Test split (80% train, 20% test)
    # stratify=y keeps the same approval ratio in both splits
    # ----------------------------------------------------------
    (X_train_raw, X_test_raw,
     y_train,     y_test,
     s_train,     s_test) = train_test_split(
        X, y, sensitive,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"\n  Train size: {len(X_train_raw)} rows")
    print(f"  Test size : {len(X_test_raw)} rows")
    print(f"  Approval rate in train: {y_train.mean():.1%}")

    # ----------------------------------------------------------
    # STEP 7: Scale numeric features with StandardScaler
    # fit on train only — never fit on test (avoids data leakage)
    # ----------------------------------------------------------
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X_train_raw.columns,
        index=X_train_raw.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X_test_raw.columns,
        index=X_test_raw.index
    )

    # ----------------------------------------------------------
    # STEP 8: Save scaler and encoders to disk
    # The Streamlit app loads these to process new user input
    # ----------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler,   "models/scaler.pkl")
    joblib.dump(encoders, "models/encoders.pkl")
    print("\n  Saved: models/scaler.pkl")
    print("  Saved: models/encoders.pkl")
    print("\nPreprocessing complete!")

    return (X_train_sc, X_test_sc,
            y_train, y_test,
            s_train, s_test,
            X_train_raw, X_test_raw)


if __name__ == "__main__":
    load_and_preprocess()