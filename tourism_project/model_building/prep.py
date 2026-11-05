"""
Data preparation step.

- Loads the dataset directly from the repository data folder
- Cleans it (drops identifier columns, fixes label typos, removes duplicates)
- Splits it into stratified train/test sets
- Saves Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv locally; the GitHub Actions
  workflow uploads them as an artifact for the model-training job.
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.getenv("DATA_PATH", "tourism_project/data/tourism.csv")
OUTPUT_DIR = os.getenv("SPLIT_DIR", ".")      # repo root, where the workflow expects the files
TARGET = "ProdTaken"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove unnecessary columns: leftover index column(s) and the customer identifier
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")] + ["CustomerID"]
    df = df.drop(columns=drop_cols, errors="ignore")
    print("Dropped columns:", drop_cols)

    # 2. Tidy text columns and fix inconsistent labels
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # 3. Rows without a target cannot be used for supervised learning
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    # 4. Remove exact duplicate records (same customer profile + outcome)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicate rows")

    # Remaining missing values (if any) are imputed inside the model pipeline,
    # using training-set statistics only, to avoid data leakage.
    return df


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print("Raw data shape:", df.shape)

    df = clean(df)
    print("Clean data shape:", df.shape)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratify to keep the ~19% purchase rate in both splits
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    Xtrain.to_csv(os.path.join(OUTPUT_DIR, "Xtrain.csv"), index=False)
    Xtest.to_csv(os.path.join(OUTPUT_DIR, "Xtest.csv"), index=False)
    ytrain.to_csv(os.path.join(OUTPUT_DIR, "ytrain.csv"), index=False)
    ytest.to_csv(os.path.join(OUTPUT_DIR, "ytest.csv"), index=False)

    print(f"Train: {Xtrain.shape}, purchase rate {ytrain.mean():.2%}")
    print(f"Test : {Xtest.shape}, purchase rate {ytest.mean():.2%}")
    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv to", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
