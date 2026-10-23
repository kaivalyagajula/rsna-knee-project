"""Resumable download of selected RSNA knee studies into data/samples.

Usage: python download_samples.py manifest.json [--retries N]

manifest.json maps StudyInstanceUID -> list of Kaggle file names
("train_series/<study>/<series>/<sop>.dcm"). Files already on disk are
skipped, failures are retried with backoff, progress prints every 50 files,
and anything still failing is listed at the end (exit code 1).
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

import kaggle

COMP = "rsna-knee-abnormality-detection"
DEST = Path("data/samples")
socket.setdefaulttimeout(60)  # a stalled read becomes a retryable error


def fetch(api, name, retries):
    """Download one file into DEST/<study>/<series>/. Return None or an error."""
    _, study, series, fname = name.split("/")
    out_dir = DEST / study / series
    out = out_dir / fname
    if out.exists() and out.stat().st_size > 0:
        return "skip"
    out_dir.mkdir(parents=True, exist_ok=True)
    err = None
    for attempt in range(retries):
        try:
            api.competition_download_file(COMP, name, path=str(out_dir), quiet=True)
            if out.exists() and out.stat().st_size > 0:
                return None
            err = "file not found after download"
        except Exception as e:  # network / API errors
            err = f"{type(e).__name__}: {e}"
        time.sleep(2 ** attempt)
    out.unlink(missing_ok=True)
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args()

    names = sorted(n for v in json.load(open(args.manifest)).values() for n in v)
    api = kaggle.api
    api.authenticate()

    done = skipped = 0
    failed = {}
    for i, name in enumerate(names, 1):
        res = fetch(api, name, args.retries)
        if res == "skip":
            skipped += 1
        elif res is None:
            done += 1
        else:
            failed[name] = res
        if i % 50 == 0 or i == len(names):
            print(f"{i}/{len(names)} downloaded={done} skipped={skipped} "
                  f"failed={len(failed)}", flush=True)

    if failed:
        print(f"\n{len(failed)} file(s) still failing:")
        for n, e in failed.items():
            print(f"  {n}: {e}")
        sys.exit(1)
    print("All files present.")


if __name__ == "__main__":
    main()
