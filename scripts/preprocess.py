"""
preprocess.py
Shrink raw NESO 1-second frequency CSVs into 1-minute files small enough for GitHub.

Input : raw monthly CSVs from NESO (columns: dtm, f), e.g. data/raw/f-2025-1.csv
        (.csv or .csv.zip both work)
Output: data/processed/<name>_1min.csv.gz  with columns
            dtm        minute timestamp (UTC)
            f_mean     mean frequency over the minute [Hz]
            f_min      minimum frequency in the minute [Hz]   (keeps sharp dips visible)
            f_max      maximum frequency in the minute [Hz]
            n_samples  number of 1-s readings in the minute    (60 = complete, <60 = gap)
        data/processed/summary.csv  one row per input file (coverage, events, etc.)

Usage:
    python preprocess.py                       # uses data/raw -> data/processed
    python preprocess.py --raw path/to/raw --out path/to/out --nominal 50
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def read_raw(path):
    """Read one NESO file and return a Series of frequency indexed by UTC timestamp."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "dtm" not in df.columns or "f" not in df.columns:
        raise ValueError(f"{path}: expected columns 'dtm' and 'f', found {list(df.columns)}")
    try:
        t = pd.to_datetime(df["dtm"], utc=True, format="ISO8601")
    except (ValueError, TypeError):
        t = pd.to_datetime(df["dtm"], utc=True, format="mixed")
    f = pd.to_numeric(df["f"], errors="coerce")
    s = pd.Series(f.values, index=t, name="f").dropna()
    s = s[~s.index.duplicated()].sort_index()
    return s


def count_events(f_min, threshold):
    """Count contiguous runs of minutes whose minimum frequency drops below threshold."""
    below = (f_min < threshold).fillna(False).to_numpy()
    starts = below & ~np.r_[False, below[:-1]]
    return int(starts.sum()), int(below.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--nominal", type=float, default=50.0, help="50 for GB/Europe, 60 for US")
    ap.add_argument("--event_dev", type=float, default=0.2,
                    help="event = minute where f_min < nominal - event_dev (Hz)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.raw, "*.csv")) +
                   glob.glob(os.path.join(args.raw, "*.csv.zip")))
    if not files:
        raise SystemExit(f"No CSV files found in {args.raw}")
    os.makedirs(args.out, exist_ok=True)
    threshold = args.nominal - args.event_dev

    rows = []
    for path in files:
        name = os.path.basename(path).replace(".csv.zip", "").replace(".csv", "")
        s = read_raw(path)
        per_min = s.resample("1min").agg(["mean", "min", "max", "count"])
        per_min.columns = ["f_mean", "f_min", "f_max", "n_samples"]
        per_min.index.name = "dtm"

        out_path = os.path.join(args.out, f"{name}_1min.csv.gz")
        per_min.round(5).to_csv(out_path)

        n_events, n_event_minutes = count_events(per_min["f_min"], threshold)
        dev = s - args.nominal
        rows.append({
            "file": name,
            "start": s.index[0], "end": s.index[-1],
            "raw_readings": len(s),
            "minutes": len(per_min),
            "complete_minutes_pct": round(100 * (per_min["n_samples"] == 60).mean(), 2),
            "empty_minutes": int((per_min["n_samples"] == 0).sum()),
            "dev_std_Hz": round(dev.std(), 4),
            "dev_min_Hz": round(dev.min(), 4),
            "dev_max_Hz": round(dev.max(), 4),
            f"events_below_{threshold:g}Hz": n_events,
            "event_minutes": n_event_minutes,
            "raw_MB": round(os.path.getsize(path) / 1e6, 1),
            "processed_MB": round(os.path.getsize(out_path) / 1e6, 2),
        })
        print(f"{name}: {len(s):,} readings -> {len(per_min):,} minutes "
              f"({rows[-1]['raw_MB']} MB -> {rows[-1]['processed_MB']} MB), "
              f"{n_events} events below {threshold:g} Hz")

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print(f"\nWrote {len(rows)} processed files and summary.csv to {args.out}")


if __name__ == "__main__":
    main()
