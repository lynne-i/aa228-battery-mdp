# Sequential State-of-Charge Management for Battery Frequency Response Under Uncertainty

AA228/CS238 Final Project (Autumn 2026) — Victoria Dinov, Lynne Irvin

Formulate battery SoC management for frequency response as an MDP, solve it with value iteration,
and compare against rule-based baselines on held-out historical frequency traces. See
[`docs/proposal.pdf`](docs/proposal.pdf).

## Layout

```
src/battery_mdp/   project code (frequency model, battery, MDP, solvers, baselines, sim, metrics)
scripts/           entry-point scripts for experiments
notebooks/         exploration / data analysis
tests/             unit tests
data/raw/          partner data (git-ignored, do not commit)
data/processed/    derived data (git-ignored)
results/figures/   generated plots
report/            status update and final report
docs/              proposal and reference material
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Milestones

- [x] Proposal
- [ ] Status update
- [ ] Final report
- [ ] Peer reviews

## Contributions

| Member | Contributions |
| --- | --- |
| Victoria Dinov | |
| Lynne Irvin | |
