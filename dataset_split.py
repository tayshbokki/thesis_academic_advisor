"""
Run this ONCE before rebuilding ChromaDB.

  - train split (80%) → ingested into ChromaDB faqs collection
  - test split  (20%) → used exclusively for evaluation

Outputs:
  dataset_train.xlsx   (used by chunking_pipeline.py)
  dataset_test.xlsx    (used by all three eval scripts)
  split_report.txt     (category breakdown for thesis documentation)

Usage:
    pip install pandas openpyxl scikit-learn
    python dataset_split.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATASET_PATH = "dataset-query.xlsx"
TRAIN_PATH   = "dataset_train.xlsx"
TEST_PATH    = "dataset_test.xlsx"
REPORT_PATH  = "split_report.txt"
TEST_SIZE    = 0.20
RANDOM_SEED  = 42


def main():
    print(f"Loading {DATASET_PATH}...")
    df = pd.read_excel(DATASET_PATH)
    print(f"  Total rows: {len(df)}")
    print(f"  Categories: {df['category'].nunique()}")

    # Drop rows missing question or answer (mirrors load_dataset() guard)
    before = len(df)
    df = df[df["question"].notna() & df["answer"].notna()].reset_index(drop=True)
    if before - len(df):
        print(f"  Dropped {before - len(df)} rows with missing question/answer.")

    # Some categories may have only 1 sample — stratified split requires ≥2.
    # Identify and handle singletons: force them into train.
    counts = df["category"].value_counts()
    singletons = counts[counts < 2].index.tolist()
    if singletons:
        print(f"\n  [WARN] {len(singletons)} category/ies with only 1 sample "
              f"(forced into train): {singletons}")
        singleton_rows = df[df["category"].isin(singletons)]
        df_splittable  = df[~df["category"].isin(singletons)]
    else:
        singleton_rows = pd.DataFrame()
        df_splittable  = df

    # Stratified split on splittable rows
    train, test = train_test_split(
        df_splittable,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df_splittable["category"],
    )

    # Append singletons to train
    if not singleton_rows.empty:
        train = pd.concat([train, singleton_rows]).reset_index(drop=True)

    test  = test.reset_index(drop=True)
    train = train.reset_index(drop=True)

    print(f"\n  Train: {len(train)} rows  ({len(train)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test)}  rows  ({len(test)/len(df)*100:.1f}%)")

    # Save splits
    train.to_excel(TRAIN_PATH, index=False)
    test.to_excel(TEST_PATH,   index=False)
    print(f"\n  Saved: {TRAIN_PATH}")
    print(f"  Saved: {TEST_PATH}")

    # Report
    lines = []
    lines.append("DATASET SPLIT REPORT")
    lines.append(f"Seed: {RANDOM_SEED} | Test size: {TEST_SIZE*100:.0f}%")
    lines.append(f"Total: {len(df)} | Train: {len(train)} | Test: {len(test)}")
    lines.append("")
    lines.append(f"{'Category':<35} {'Total':>6} {'Train':>6} {'Test':>6}")
    lines.append("-" * 57)

    all_cats = sorted(df["category"].unique())
    for cat in all_cats:
        total = len(df[df["category"] == cat])
        tr    = len(train[train["category"] == cat])
        te    = len(test[test["category"] == cat])
        lines.append(f"{cat:<35} {total:>6} {tr:>6} {te:>6}")

    report = "\n".join(lines)
    print("\n" + report)
    Path(REPORT_PATH).write_text(report, encoding="utf-8")
    print(f"\n  Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
