"""Show 4 evenly spaced slices from one DICOM series.

Usage: python view_series.py data/samples/<StudyInstanceUID>/<SeriesInstanceUID>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pydicom

# Compressed transfer syntaxes need a pixel-data handler. Import errors are
# reported at decode time, with a hint on what to install.
HINT = (
    "Could not decode pixel data. For compressed DICOM install: "
    "pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm pillow"
)


def load_series(series_dir):
    files = sorted(Path(series_dir).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm files in {series_dir}")
    headers = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]

    def key(item):
        ds = item[1]
        pos = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        if pos is not None and iop is not None:
            normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
            return float(np.dot(normal, np.array(pos, float)))
        return float(getattr(ds, "InstanceNumber", 0) or 0)

    order = sorted(zip(files, headers), key=key)
    return [f for f, _ in order]


def read_pixels(path):
    ds = pydicom.dcmread(path)
    try:
        arr = ds.pixel_array.astype(np.float32)
    except Exception as e:  # missing handler for the file's compression
        raise RuntimeError(
            f"{HINT}\nTransfer syntax: {ds.file_meta.TransferSyntaxUID.name}\n{e}"
        ) from e
    if arr.ndim == 3 and arr.shape[-1] in (3, 4) and \
            getattr(ds, "SamplesPerPixel", 1) > 1:
        arr = arr[..., :3].mean(-1)  # RGB -> gray
    elif arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]  # multi-frame: take middle frame
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    files = load_series(sys.argv[1])
    idx = np.linspace(0, len(files) - 1, 4).round().astype(int)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, i in zip(axes, idx):
        img = read_pixels(files[i])
        lo, hi = np.percentile(img, [1, 99])
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(f"slice {i + 1}/{len(files)}")
        ax.axis("off")
    fig.suptitle(Path(sys.argv[1]).name, fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
