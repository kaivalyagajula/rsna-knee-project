"""Pick the series to use for a study from train_series.csv.

Usage: python select_series.py [StudyInstanceUID ...]   (default: the studies in data/samples)
"""
import sys
from pathlib import Path

import pandas as pd

SERIES_CSV = Path("data/train_series.csv")
PLANES = ("Sagittal", "Coronal", "Axial")


def select_series(study_uid, series_df=None):
    """Return {plane: SeriesInstanceUID} for the best series of each plane.

    Within a plane, prefer Fluid_Sensitive=1, then Fat_Suppression=1, then the
    lowest series UID so the choice is deterministic. Planes with no series are
    left out; Sagittal is the one the caller should normally expect.
    """
    if series_df is None:
        series_df = pd.read_csv(SERIES_CSV)
    rows = series_df[series_df.StudyInstanceUID == study_uid]
    picked = {}
    for plane in PLANES:
        cand = rows[rows.Anatomical_Plane == plane]
        if cand.empty:
            continue
        best = cand.sort_values(
            ["Fluid_Sensitive", "Fat_Suppression", "SeriesInstanceUID"],
            ascending=[False, False, True],
        ).iloc[0]
        picked[plane] = best.SeriesInstanceUID
    return picked


def main():
    studies = sys.argv[1:] or sorted(p.name for p in Path("data/samples").iterdir() if p.is_dir())
    df = pd.read_csv(SERIES_CSV)
    for study in studies:
        rows = df[df.StudyInstanceUID == study]
        print(f"{study}  ({len(rows)} series)")
        picked = select_series(study, df)
        for plane in PLANES:
            if plane not in picked:
                print(f"  {plane:9s} -")
                continue
            r = rows[rows.SeriesInstanceUID == picked[plane]].iloc[0]
            print(f"  {plane:9s} {picked[plane]}  fluid={r.Fluid_Sensitive} fat_sup={r.Fat_Suppression}")


if __name__ == "__main__":
    main()
