"""
make_synthetic_data.py
Create fake NESO-format files (columns dtm, f at 1-second resolution) to test the pipeline.
NOT real data: use only to check that preprocess.py and frequency_mdp.py run end to end.

Model: Ornstein-Uhlenbeck deviation around 50 Hz, a slow random drift term, and a few
injected contingency-like dips (sudden drop, then recovery over a few minutes).

Usage: python make_synthetic_data.py --days 14 --files 2 --out data/raw_synthetic
"""
import argparse
import os

import numpy as np
import pandas as pd


def make_month(start, days, rng, nominal=50.0):
    n = int(days * 86400)
    theta, sigma = 1 / 120.0, 0.0045          # reversion ~2 min; stationary std ~0.04 Hz
    x = np.empty(n); x[0] = 0.0
    noise = rng.normal(size=n)
    for t in range(1, n):
        x[t] = x[t - 1] - theta * x[t - 1] + sigma * noise[t]
    drift = 0.02 * np.sin(2 * np.pi * np.arange(n) / 3600.0 * rng.uniform(0.5, 2.0))
    f = nominal + x + drift
    for _ in range(max(1, days // 4)):          # inject contingency-like dips
        t0 = rng.integers(0, n - 900)
        depth = rng.uniform(0.25, 0.45)
        k = np.arange(900)
        f[t0:t0 + 900] -= depth * np.exp(-k / 180.0) * (1 - np.exp(-k / 3.0))
    idx = pd.date_range(start, periods=n, freq="1s")
    return pd.DataFrame({"dtm": idx.strftime("%Y-%m-%d %H:%M:%S"), "f": np.round(f, 3)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--files", type=int, default=2)
    ap.add_argument("--out", default="data/raw_synthetic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    start = pd.Timestamp("2025-01-01")
    for i in range(args.files):
        df = make_month(start, args.days, rng)
        path = os.path.join(args.out, f"synthetic-{i + 1}.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path} ({len(df):,} rows)")
        start += pd.Timedelta(days=args.days)


if __name__ == "__main__":
    main()
