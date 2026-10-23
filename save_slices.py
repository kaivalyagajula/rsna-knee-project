"""Save a 4-slice figure for the first series of each study in data/samples.

Usage: python save_slices.py  ->  outputs/slices_study<N>.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from view_series import load_series, read_pixels


def main():
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    studies = sorted(p for p in Path("data/samples").iterdir() if p.is_dir())
    for n, study in enumerate(studies, 1):
        series = sorted(p for p in study.iterdir() if p.is_dir())[0]
        files = load_series(series)
        idx = np.linspace(0, len(files) - 1, 4).round().astype(int)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        for ax, i in zip(axes, idx):
            img = read_pixels(files[i])
            lo, hi = np.percentile(img, [1, 99])
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
            ax.set_title(f"slice {i + 1}/{len(files)}")
            ax.axis("off")
        fig.suptitle(f"study {n}: {study.name} / {series.name}", fontsize=7)
        plt.tight_layout()
        fig.savefig(out / f"slices_study{n}.png", dpi=100)
        plt.close(fig)
        print(f"saved outputs/slices_study{n}.png")


if __name__ == "__main__":
    main()
