# Battery SoC management for frequency response (AA228/CS238 final project)

## Data
Raw data: NESO (Great Britain) historic system frequency, 1-second resolution, monthly CSVs
(columns `dtm`, `f`), from the NESO Data Portal ("System Frequency" dataset).
Supported by National Grid ESO Open Data.

Months used: _list the files here, e.g. f-2025-1.csv, f-2025-2.csv, f-2025-3.csv_

Raw files are **not** in this repo (too large). Put them in `data/raw/` locally.
`data/processed/` holds the 1-minute versions produced by `preprocess.py` (small enough to commit).

## Run
```bash
pip install -r requirements.txt
python preprocess.py                 # data/raw -> data/processed (+ summary.csv)
python frequency_mdp.py              # data/processed -> results/
```
Options: `--nominal 60` for US data, `--test_frac 0.25` (last 25% of time held out),
`--gamma 0.9999` (per-minute discount, ~1 week horizon).

Pipeline test without real data:
```bash
python make_synthetic_data.py --days 14 --files 2
python preprocess.py --raw data/raw_synthetic --out data/processed_synthetic
python frequency_mdp.py --data data/processed_synthetic --out results_synthetic --test_frac 0.5
```

## Outputs (`results/`)
- `results_test.csv`: metrics for each policy on the held-out test trace
- `transition_matrix.csv`: estimated T(f' | f)
- `run_info.json`: data periods, bin counts, all parameters used
- `fig_data.png`, `fig_policy.png`, `fig_soc_test.png`

## Model summary
State = (SoC on a 51-point grid, bin of last minute's mean frequency deviation; 9 bins).
Action = SoC-management power in {-0.2, ..., 0.2} MW. One step = 1 minute.
Droop response (deadband 0.015 Hz, full 1 MW at 0.5 Hz) is served first; management power after.
Reward = -(100 * shortfall MWh + 1 * throughput MWh + 0.5 * management charging MWh).
T(f' | f) is the Dirichlet posterior mean (alpha = 1) from training data. Solved by value
iteration; evaluated by simulating a continuous-SoC battery on the real held-out trace.
Baselines: no management, proportional controller to 50% SoC, myopic (gamma = 0).
All parameters are at the top of `frequency_mdp.py`.
