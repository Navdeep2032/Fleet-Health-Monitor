"""
train_model.py

Week 3 (ML) core step. Loads data/fleet_training_data.csv (produced by
generate_training_data.py), builds rolling-window features per (run_id,
cycle), trains a RandomForestClassifier to predict fault_type, prints a
classification report, and saves the model to models/fault_classifier.pkl.

Run from the package root (after generate_training_data.py):
    python3 scripts/train_model.py
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fleet_health_monitor.feature_engineering import compute_features, WINDOW, FEATURE_NAMES

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'fleet_training_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fault_classifier.pkl')

# Only build a feature row every STRIDE cycles per run (keeps dataset size
# sane; rolling windows overlap heavily otherwise).
STRIDE = 5


def build_feature_table(df: pd.DataFrame):
    X, y = [], []
    for run_id, run_df in df.groupby('run_id'):
        run_df = run_df.sort_values('cycle').reset_index(drop=True)
        fault_type = run_df['fault_type'].iloc[0]

        current = run_df['current'].values
        vibration = run_df['vibration'].values
        temp = run_df['temp'].values

        for end in range(WINDOW, len(run_df), STRIDE):
            start = end - WINDOW
            feats = compute_features(
                current[start:end], vibration[start:end], temp[start:end])
            X.append(feats)
            y.append(fault_type)

    return np.array(X), np.array(y)


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found. Run generate_training_data.py first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} raw rows across {df['run_id'].nunique()} runs.")

    X, y = build_feature_table(df)
    print(f"Built feature table: {X.shape[0]} samples x {X.shape[1]} features.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n=== Classification report (held-out test set) ===")
    print(classification_report(y_test, y_pred))
    print("=== Confusion matrix ===")
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("labels:", labels)
    print(cm)

    print("\n=== Feature importances ===")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"  {name:20s} {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == '__main__':
    main()
