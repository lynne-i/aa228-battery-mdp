"""
frequency_mdp.py
Battery state-of-charge (SoC) management for frequency response, formulated as an MDP.

Pipeline
  1. Load processed 1-minute files (from preprocess.py); split chronologically into train/test.
  2. Discretize frequency deviation into bins; estimate T(f' | f) from TRAIN data with a
     Dirichlet prior (posterior mean = (counts + alpha) / row total).
  3. Solve the MDP with value iteration.
  4. Evaluate the VI policy and baselines by simulating a continuous-SoC battery on the
     held-out TEST trace (real data, not the learned model).
  5. Save a results table, a text summary, and figures to results/.

MDP (one step = 1 minute)
  State   s_t = (e_t, f_t): SoC level on a grid, and the bin of the last observed minute's
          mean frequency deviation.
  Action  u_t: SoC-management power [MW] (positive = charge), from a small discrete set.
  Step    during minute t+1 the battery must deliver droop response for the realized
          deviation (bin f_{t+1} ~ T(.|f_t)); then u_t is applied. Response has priority.
  Reward  -lam_short * shortfall_MWh - lam_deg * throughput_MWh - lam_energy * charged_MWh
          (flat energy cost; no prices)
  Because the step uses the *next* minute's deviation, the policy never sees the future:
  the same causal timing is used in the simulator.
  Off-grid SoC values are handled by linear interpolation between neighbouring grid points
  (probability split), which preserves expected SoC even when one-minute energy changes are
  smaller than the grid spacing.

Usage
  python frequency_mdp.py                         # data/processed -> results/
  python frequency_mdp.py --test_frac 0.25 --gamma 0.9999
  python frequency_mdp.py --nominal 60            # US data
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# Parameters (all in one place so they are easy to report and to sweep later)
# --------------------------------------------------------------------------------------
BATTERY = dict(
    E_cap=1.0,        # usable energy capacity [MWh]
    E_min_frac=0.0,   # SoC lower bound as fraction of E_cap
    E_max_frac=1.0,   # SoC upper bound
    P_commit=1.0,     # committed frequency-response capacity [MW]
    eta_c=0.95,       # charging efficiency  (round trip ~0.90)
    eta_d=0.95,       # discharging efficiency
)
DROOP = dict(
    deadband=0.015,   # [Hz] no response inside +/- deadband
    full_dev=0.5,     # [Hz] deviation at which full P_commit is delivered (linear in between)
)
ACTIONS_MW = np.array([-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2])  # SoC-management power
DEV_EDGES = np.array([-np.inf, -0.2, -0.1, -0.05, -0.015, 0.015, 0.05, 0.1, 0.2, np.inf])
N_SOC = 51          # SoC grid points
DT_H = 1.0 / 60.0   # step length [h]
REWARD = dict(lam_short=100.0, lam_deg=1.0, lam_energy=0.5)
ALPHA = 1.0         # Dirichlet pseudo-count per transition cell
NEAR_LIMIT_FRAC = 0.05


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_processed(folder, nominal):
    files = sorted(glob.glob(os.path.join(folder, "*_1min.csv.gz")))
    if not files:
        raise SystemExit(f"No *_1min.csv.gz files in {folder}. Run preprocess.py first.")
    df = pd.concat([pd.read_csv(p, parse_dates=["dtm"], index_col="dtm") for p in files])
    df = df[~df.index.duplicated()].sort_index()
    df["dev"] = df["f_mean"] - nominal
    return df, files


def bin_of(dev):
    """Bin index for each deviation (NaN -> -1)."""
    b = np.digitize(dev, DEV_EDGES[1:-1])
    return np.where(np.isnan(dev), -1, b)


def estimate_transitions(dev):
    """Count transitions between consecutive *valid* minutes and apply Dirichlet prior."""
    b = bin_of(dev)
    F = len(DEV_EDGES) - 1
    counts = np.zeros((F, F))
    ok = (b[:-1] >= 0) & (b[1:] >= 0)
    np.add.at(counts, (b[:-1][ok], b[1:][ok]), 1)
    T = (counts + ALPHA) / (counts + ALPHA).sum(axis=1, keepdims=True)
    # representative deviation per bin = empirical mean in training data
    rep = np.zeros(F)
    for k in range(F):
        vals = dev[b == k]
        lo, hi = DEV_EDGES[k], DEV_EDGES[k + 1]
        rep[k] = vals.mean() if len(vals) else np.clip(0.0, max(lo, -0.5), min(hi, 0.5))
    return T, counts, rep


# --------------------------------------------------------------------------------------
# Battery physics (shared by the MDP model and the simulator)
# --------------------------------------------------------------------------------------
def droop_power(dev):
    """Required response power [MW], positive = charge. Under-frequency -> discharge."""
    db, full, P = DROOP["deadband"], DROOP["full_dev"], BATTERY["P_commit"]
    mag = np.clip((np.abs(dev) - db) / (full - db), 0.0, 1.0) * P
    return np.sign(dev) * mag


def step_battery(e, dev, u):
    """
    One minute of operation. Response is served first, then the management power u.
    Returns next SoC, shortfall [MWh of response not delivered], throughput [MWh], charged [MWh].
    Works elementwise on numpy arrays.
    """
    E_lo = BATTERY["E_min_frac"] * BATTERY["E_cap"]
    E_hi = BATTERY["E_max_frac"] * BATTERY["E_cap"]
    ec, ed = BATTERY["eta_c"], BATTERY["eta_d"]

    def apply(e, p):
        # energy change for power p over DT_H, limited by SoC bounds; returns new e and
        # the grid-side energy actually exchanged (signed, positive = charge)
        want = np.where(p > 0, ec * p * DT_H, p * DT_H / ed)          # change in stored energy
        e_new = np.clip(e + want, E_lo, E_hi)
        stored = e_new - e
        grid = np.where(stored > 0, stored / ec, stored * ed)          # grid-side energy
        return e_new, grid

    r = droop_power(dev)
    e1, grid_r = apply(e, r)
    shortfall = np.abs(r * DT_H - grid_r)
    e2, grid_u = apply(e1, u)
    throughput = np.abs(grid_r) + np.abs(grid_u)
    charged = np.maximum(grid_u, 0.0)
    return e2, shortfall, throughput, charged


def reward(shortfall, throughput, charged):
    return -(REWARD["lam_short"] * shortfall + REWARD["lam_deg"] * throughput
             + REWARD["lam_energy"] * charged)


# --------------------------------------------------------------------------------------
# MDP solution
# --------------------------------------------------------------------------------------
def soc_grid():
    return np.linspace(BATTERY["E_min_frac"], BATTERY["E_max_frac"], N_SOC) * BATTERY["E_cap"]


def build_model(rep):
    """For every (e, a, f'): next-SoC interpolation indices/weights and immediate reward."""
    grid = soc_grid()
    E, A, F = N_SOC, len(ACTIONS_MW), len(rep)
    e = grid[:, None, None] * np.ones((1, A, F))
    u = ACTIONS_MW[None, :, None] * np.ones((E, 1, F))
    dev = rep[None, None, :] * np.ones((E, A, 1))
    e_next, short, thr, chg = step_battery(e, dev, u)
    R = reward(short, thr, chg)                                         # (E, A, F')
    pos = (e_next - grid[0]) / (grid[1] - grid[0])
    lo = np.clip(np.floor(pos).astype(int), 0, E - 2)
    w_hi = np.clip(pos - lo, 0.0, 1.0)
    return R, lo, w_hi


def value_iteration(T, R, lo, w_hi, gamma, tol=1e-7, max_iter=200_000):
    E, A, F = R.shape
    V = np.zeros((E, F))
    f_idx = np.arange(F)[None, None, :]
    for it in range(max_iter):
        Vn = (1 - w_hi) * V[lo, f_idx] + w_hi * V[lo + 1, f_idx]      # V(e', f') : (E, A, F')
        G = R + gamma * Vn                                              # value of each outcome
        Q = np.einsum("eaj,fj->efa", G, T)                              # expectation over f'
        V_new = Q.max(axis=2)
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol * (1 - gamma) / max(gamma, 1e-12):
            break
    policy = Q.argmax(axis=2)                                           # (E, F) action index
    return V, Q, policy, it + 1


# --------------------------------------------------------------------------------------
# Policies and simulation on real data
# --------------------------------------------------------------------------------------
def make_table_policy(policy):
    grid = soc_grid()
    def act(e, f_bin):
        i = int(np.clip(np.rint((e - grid[0]) / (grid[1] - grid[0])), 0, N_SOC - 1))
        return ACTIONS_MW[policy[i, f_bin]]
    return act


def make_proportional_policy(target_frac=0.5, gain=1.0):
    """Steer SoC toward a target: u = gain * (target - e) / E_cap  [MW], snapped to action set."""
    def act(e, f_bin):
        u = gain * (target_frac * BATTERY["E_cap"] - e) / BATTERY["E_cap"]
        return ACTIONS_MW[np.argmin(np.abs(ACTIONS_MW - u))]
    return act


def simulate(dev, act, e0_frac=0.5):
    """Continuous-SoC simulation over a real deviation trace. Missing minutes -> no response."""
    b = bin_of(dev)
    dev = np.nan_to_num(dev, nan=0.0)
    n = len(dev)
    e = e0_frac * BATTERY["E_cap"]
    last_bin = int(np.digitize(0.0, DEV_EDGES[1:-1]))
    soc = np.empty(n); short = np.empty(n); thr = np.empty(n); chg = np.empty(n); act_log = np.empty(n)
    for t in range(n):
        u = act(e, last_bin)                       # decide using last observed minute only
        e, s, th, c = step_battery(np.array(e), np.array(dev[t]), np.array(u))
        e = float(e)
        soc[t], short[t], thr[t], chg[t], act_log[t] = e, float(s), float(th), float(c), u
        if b[t] >= 0:
            last_bin = int(b[t])
    return dict(soc=soc, short=short, thr=thr, chg=chg, u=act_log)


def metrics(sim, dev):
    req = np.abs(droop_power(np.nan_to_num(dev))) > 0
    full = (sim["short"] < 1e-9) | ~req
    E = BATTERY["E_cap"]
    lo = BATTERY["E_min_frac"] * E; hi = BATTERY["E_max_frac"] * E
    near = (sim["soc"] < lo + NEAR_LIMIT_FRAC * E) | (sim["soc"] > hi - NEAR_LIMIT_FRAC * E)
    r = reward(sim["short"], sim["thr"], sim["chg"])
    return {
        "full_response_pct": round(100 * full.mean(), 3),
        "response_minutes_with_shortfall": int((~full).sum()),
        "shortfall_MWh": round(sim["short"].sum(), 4),
        "throughput_MWh": round(sim["thr"].sum(), 3),
        "equiv_full_cycles": round(sim["thr"].sum() / (2 * E), 3),
        "mgmt_energy_charged_MWh": round(sim["chg"].sum(), 3),
        "time_near_limits_pct": round(100 * near.mean(), 2),
        "final_soc_frac": round(sim["soc"][-1] / E, 3),
        "total_reward": round(r.sum(), 2),
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def bin_labels():
    labs = []
    for k in range(len(DEV_EDGES) - 1):
        lo, hi = DEV_EDGES[k], DEV_EDGES[k + 1]
        labs.append(f"<{hi:g}" if np.isinf(lo) else (f">{lo:g}" if np.isinf(hi) else f"[{lo:g},{hi:g})"))
    return labs


def make_figures(out, dev_train, T, policy, sims, test_index):
    labs = bin_labels()

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(dev_train[~np.isnan(dev_train)], bins=200, color="tab:blue")
    ax[0].set_yscale("log"); ax[0].set_xlabel("1-min mean deviation [Hz]"); ax[0].set_ylabel("count (log)")
    ax[0].set_title("Frequency deviation (train)")
    im = ax[1].imshow(T, cmap="viridis", vmin=0, vmax=1)
    ax[1].set_xticks(range(len(labs))); ax[1].set_xticklabels(labs, rotation=60, fontsize=7)
    ax[1].set_yticks(range(len(labs))); ax[1].set_yticklabels(labs, fontsize=7)
    ax[1].set_xlabel("next bin f'"); ax[1].set_ylabel("current bin f")
    ax[1].set_title("Estimated T(f' | f)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_data.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    grid = soc_grid() / BATTERY["E_cap"]
    im = ax.imshow(ACTIONS_MW[policy], aspect="auto", origin="lower", cmap="coolwarm",
                   extent=[-0.5, len(labs) - 0.5, grid[0], grid[-1]],
                   vmin=ACTIONS_MW.min(), vmax=ACTIONS_MW.max())
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=60, fontsize=7)
    ax.set_xlabel("last observed deviation bin [Hz]"); ax.set_ylabel("SoC (fraction)")
    ax.set_title("Value-iteration policy: management power u [MW]")
    fig.colorbar(im, ax=ax, label="u [MW] (+ = charge)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_policy.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    styles = ["-", "--", "-.", ":"]
    for k, (name, sim) in enumerate(sims.items()):
        ax.plot(test_index, sim["soc"] / BATTERY["E_cap"], label=name, lw=1.2,
                ls=styles[k % len(styles)])
    ax.set_ylabel("SoC (fraction)"); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8, ncol=4)
    ax.set_title("SoC on held-out test trace")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_soc_test.png"), dpi=200); plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--nominal", type=float, default=50.0)
    ap.add_argument("--test_frac", type=float, default=0.25, help="last fraction of time used as test")
    ap.add_argument("--gamma", type=float, default=0.9999,
                    help="per-minute discount; 0.9999 ~ 1-week effective horizon")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, files = load_processed(args.data, args.nominal)
    split = int(len(df) * (1 - args.test_frac))
    train, test = df.iloc[:split], df.iloc[split:]
    dev_tr, dev_te = train["dev"].to_numpy(), test["dev"].to_numpy()
    print(f"Loaded {len(files)} file(s): {len(df):,} minutes "
          f"({df.index[0]} to {df.index[-1]}); train {len(train):,}, test {len(test):,}")

    T, counts, rep = estimate_transitions(dev_tr)
    R, lo, w_hi = build_model(rep)
    V, Q, policy, iters = value_iteration(T, R, lo, w_hi, args.gamma)
    print(f"Value iteration converged in {iters:,} iterations "
          f"({N_SOC * len(rep)} states, {len(ACTIONS_MW)} actions)")
    _, _, myopic, _ = value_iteration(T, R, lo, w_hi, gamma=0.0)

    policies = {
        "Value iteration": make_table_policy(policy),
        "No management": lambda e, f: 0.0,
        "Proportional (50% target)": make_proportional_policy(0.5),
        "Myopic (gamma=0)": make_table_policy(myopic),
    }
    sims, rows = {}, []
    for name, act in policies.items():
        sims[name] = simulate(dev_te, act)
        rows.append({"policy": name, **metrics(sims[name], dev_te)})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "results_test.csv"), index=False)
    pd.DataFrame(T, index=bin_labels(), columns=bin_labels()).round(4).to_csv(
        os.path.join(args.out, "transition_matrix.csv"))

    make_figures(args.out, dev_tr, T, policy, sims, test.index)

    info = dict(files=[os.path.basename(f) for f in files],
                minutes_total=len(df), train_minutes=len(train), test_minutes=len(test),
                train_period=[str(train.index[0]), str(train.index[-1])],
                test_period=[str(test.index[0]), str(test.index[-1])],
                missing_minutes_test=int(np.isnan(dev_te).sum()),
                transitions_counted=int(counts.sum()),
                bin_counts_train=dict(zip(bin_labels(), np.bincount(bin_of(dev_tr)[bin_of(dev_tr) >= 0],
                                                                     minlength=len(rep)).tolist())),
                vi_iterations=iters, gamma=args.gamma,
                battery=BATTERY, droop=DROOP, reward=REWARD, actions_MW=ACTIONS_MW.tolist(),
                n_soc=N_SOC, dirichlet_alpha=ALPHA)
    with open(os.path.join(args.out, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)

    pd.set_option("display.width", 200)
    print("\nHeld-out test results:")
    print(table.to_string(index=False))
    print(f"\nSaved results, figures, and run_info.json to {args.out}/")


if __name__ == "__main__":
    main()
