"""
preprocess.py
-------------
Data loading, cleaning, and feature engineering for the
CodeAlpha Credit Scoring Model project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ── Column names for the raw UCI german.data file (space-separated, no header)
UCI_COLUMNS = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment", "installment_rate",
    "personal_status", "other_debtors", "residence_since", "property",
    "age", "other_installments", "housing", "existing_credits",
    "job", "liable_people", "telephone", "foreign_worker", "risk"
]

CATEGORICAL_COLS = [
    "checking_account", "credit_history", "purpose", "savings_account",
    "employment", "personal_status", "other_debtors", "property",
    "other_installments", "housing", "job", "telephone", "foreign_worker"
]

NUMERIC_COLS = [
    "duration", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "liable_people"
]


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the German Credit dataset.
    Supports:
      - Kaggle CSV  (has a header row, semicolon or comma separated)
      - UCI raw     (no header, space separated — german.data)
    """
    # ── Try reading as a standard CSV first (Kaggle version)
    try:
        df = pd.read_csv(filepath)
        # Kaggle file uses 'Risk' column with Good/Bad strings
        if "Risk" in df.columns:
            df = df.rename(columns={"Risk": "risk"})
            df["risk"] = df["risk"].map({"good": 0, "bad": 1})
        print(f"[load] Loaded Kaggle CSV — {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception:
        pass

    # ── Fall back to raw UCI format
    df = pd.read_csv(filepath, sep=" ", header=None, names=UCI_COLUMNS)
    # UCI: 1 = Good, 2 = Bad  →  remap to 0 / 1
    df["risk"] = df["risk"].map({1: 0, 2: 1})
    print(f"[load] Loaded UCI raw file — {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and fill missing values."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"[clean] Dropped {before - len(df)} duplicate rows")

    # Fill numeric NaNs with median, categorical with mode
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)

    print(f"[clean] Missing values remaining: {df.isnull().sum().sum()}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from existing columns."""
    df = df.copy()

    # These columns exist in both Kaggle and UCI versions (after renaming)
    if "credit_amount" in df.columns and "duration" in df.columns:
        df["monthly_payment"] = df["credit_amount"] / df["duration"].replace(0, np.nan)

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 50, 100],
            labels=["young", "adult", "middle_aged", "senior"]
        ).astype(str)

    if "credit_amount" in df.columns and "age" in df.columns:
        df["credit_per_age"] = df["credit_amount"] / df["age"].replace(0, np.nan)

    print("[feature_engineering] New features added: monthly_payment, age_group, credit_per_age")
    return df


def encode_and_scale(df: pd.DataFrame):
    """
    Label-encode all object columns, then standard-scale numeric columns.
    Returns (X_scaled, y, scaler, encoders_dict).
    """
    df = df.copy()
    target = "risk"

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Check your dataset.")

    y = df[target].values
    X = df.drop(columns=[target])

    # Encode categoricals
    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Scale numerics
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    print(f"[encode_and_scale] Features: {X.shape[1]}  |  Target distribution: {pd.Series(y).value_counts().to_dict()}")
    return X, y, scaler, encoders


def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[split] Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test
