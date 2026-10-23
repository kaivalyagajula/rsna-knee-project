"""Summarize label coverage in train.csv and print sample radiology reports."""

import argparse
import textwrap
from pathlib import Path

import pandas as pd

REPORT_COL = "Report"
LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--n-reports", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {args.csv}: {len(df):,} studies")

    counts = df[LABEL_COLS].notna().sum()
    summary = pd.DataFrame({
        "non_null": counts,
        "pct": (counts / len(df) * 100).round(1),
    })
    print("\nNon-null studies per label column:")
    print(summary.to_string())
    n_any = df[LABEL_COLS].notna().any(axis=1).sum()
    print(f"\nStudies with at least one label: {n_any:,} ({n_any / len(df):.1%})")

    reports = df[df[REPORT_COL].notna()]
    sample = reports.sample(n=min(args.n_reports, len(reports)), random_state=args.seed)
    print(f"\n{len(sample)} example radiology reports:")
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"\n--- Report {i} ({row['StudyInstanceUID']}) ---")
        print(textwrap.fill(" ".join(str(row[REPORT_COL]).split()), width=100))


if __name__ == "__main__":
    main()
