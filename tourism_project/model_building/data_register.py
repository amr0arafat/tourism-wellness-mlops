"""
Data registration step.

The dataset lives inside the GitHub repository (tourism_project/data/tourism.csv),
so "registering" it means: load it from the repo, validate that it has the expected
schema, and print a short summary. Any validation failure exits with a non-zero
code, which stops the GitHub Actions pipeline before bad data reaches training.
"""
import os
import sys

import pandas as pd

DATA_PATH = os.getenv("DATA_PATH", "tourism_project/data/tourism.csv")
TARGET = "ProdTaken"

# Columns described in the data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]


def main() -> None:
    # 1. The file must exist in the repository
    if not os.path.exists(DATA_PATH):
        sys.exit(f"[ERROR] Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # 2. All expected columns must be present
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        sys.exit(f"[ERROR] Missing expected columns: {missing_cols}")
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    # 3. The target must be binary (0/1) and the file must not be empty
    if df.empty:
        sys.exit("[ERROR] Dataset is empty")
    bad_target = set(df[TARGET].dropna().unique()) - {0, 1}
    if bad_target:
        sys.exit(f"[ERROR] Unexpected target values: {bad_target}")

    # 4. Summary
    print("=" * 60)
    print("DATASET REGISTRATION SUMMARY")
    print("=" * 60)
    print(f"Source file        : {DATA_PATH}")
    print(f"Rows x Columns     : {df.shape[0]} x {df.shape[1]}")
    print(f"Expected columns   : all {len(EXPECTED_COLUMNS)} present")
    print(f"Extra columns      : {extra_cols if extra_cols else 'none'}")
    print(f"Duplicate CustomerID: {df['CustomerID'].duplicated().sum()}")
    print(f"Total missing cells: {int(df.isna().sum().sum())}")
    print("\nTarget distribution (ProdTaken):")
    print(df[TARGET].value_counts().rename(index={0: "0 - Not purchased", 1: "1 - Purchased"}).to_string())
    print(f"Purchase rate      : {df[TARGET].mean():.2%}")
    print("\nColumn types:")
    print(df[EXPECTED_COLUMNS].dtypes.to_string())
    print("\n[OK] Dataset validated and registered.")


if __name__ == "__main__":
    main()
