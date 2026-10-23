"""Turn a selected DICOM series into a normalized tensor.

Usage: python preprocess.py   (tests every selected series of the sample studies)
"""
from pathlib import Path

import torch
import torch.nn.functional as F

from select_series import select_series
from view_series import load_series, read_pixels

SIZE = 224
SAMPLES = Path("data/samples")


def preprocess_series(series_dir, size=SIZE):
    """Return a float32 tensor of shape (slices, size, size).

    Slices are sorted by position along the slice normal, resized to
    size x size, then the volume is clipped to its 1st-99th percentile and
    z-scored (mean 0, std 1) so scanner intensity scales do not matter.
    """
    slices = []
    for f in load_series(series_dir):
        t = torch.from_numpy(read_pixels(f))[None, None]
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        slices.append(t[0, 0])
    vol = torch.stack(slices)
    lo, hi = torch.quantile(vol.flatten(), torch.tensor([0.01, 0.99]))
    vol = vol.clamp(lo.item(), hi.item())
    return (vol - vol.mean()) / (vol.std() + 1e-6)


def main():
    for study in sorted(p for p in SAMPLES.iterdir() if p.is_dir()):
        print(study.name)
        for plane, series in select_series(study.name).items():
            x = preprocess_series(study / series)
            print(f"  {plane:9s} shape={tuple(x.shape)} mean={x.mean():.3f} std={x.std():.3f}")


if __name__ == "__main__":
    main()
