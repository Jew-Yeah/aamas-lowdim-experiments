[English](README.md) | [Русский](README.ru.md)

# Low-dimensional opponent games: experiments

Reproducible experiments for vector-payoff games in which a learner selects an
action before observing the current opponent action. The repository implements
the one-switch learner, its explicit block-safe routine, the shared past-hull
policy related to Marinov et al. (2026), and interpretable allocation heuristics.

One application allocates service quotas across five New York City boroughs
using public Forestry Hazard request counts. This is a daily allocation model.
The finite game uses demand profiles fitted on 2019; 2021–2022 are held out.
Service metrics are also evaluated on the original daily counts, separately
from target distances in the quantized game.

A second application selects network-defense policies in the official CAGE 2
simulator against changing attack modes. The recorded experiment includes
1,440 simulator episodes and compares 13 methods. See the
[CAGE results](results/cage_reference/README.md) and
[reproduction instructions](docs/cage_en.md).

The [extended CAGE study](results/cage_study/README.md) adds an independent
test bank, more attacker paths, paired statistical comparisons, and checks
of episode lengths and calibration sensitivity.

The [oracle tuning and restart study](results/cage_adaptation/README.md)
uses separate validation and final-test seeds to compare a scalar-aware
admissible saddle oracle, tuned window and Hedge baselines, and both
fresh-history and retained-history restarts in blocks of 1000 meta-rounds.
See the [methodology](docs/cage_adaptation_en.md) and
[segment bounds](docs/cage_restart_theory_en.md).

## Quick start

Python 3.10 or later is required. Use a virtual environment.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
python -m lowdim_games.cli synthetic --q 1 2 --horizon 32 --seeds 42 --plots
```

## Experiments

The repository includes aggregate public-data inputs and their provenance.
To verify or reuse the cached historical selection (add `--refresh` to fetch a new snapshot):

```bash
python scripts/download_nyc.py --years 2019 2021 2022
python -m lowdim_games.cli nyc --profiles 3 6 12 --capacity-ratios 0.6 0.8 1.0
python -m lowdim_games.cli synthetic --q 1 2 3 4 5 --horizon 128 --seeds 20261003 20261004 20261005
python -m lowdim_games.cli geometry --q 1 2 3 4 5 --horizons 32 64 128
```

Outputs include JSON configurations, solver diagnostics, learner actions,
opponent actions, and PNG/PDF figures. Default outputs are in `results/runs/`.
The [recorded reference run](results/reference_run/README.md) describes the
experiments actually completed and their findings.

## Documentation

- [Mathematical algorithms and numerical checks](docs/algorithms_en.md)
- [Data selection and preprocessing](docs/data_en.md)
- [Experimental protocol and interpretation](docs/methodology_en.md)
- [Reference results](results/reference_run/README.md)
- [CAGE 2 defense against changing attackers](docs/cage_en.md)
- [CAGE 2 recorded results](results/cage_reference/README.md)
- [Extended CAGE protocol and statistics](docs/cage_study_en.md)
- [Extended CAGE results](results/cage_study/README.md)

The target oracle evaluates the response map over the full realized opponent
hull, including responses at previously unobserved mixtures. Numerical gaps
are reported; they provide floating-point accuracy checks rather than formal
exact-arithmetic certification. The one-switch threshold follows the theorem
without empirical tuning. For short horizons it may never trigger, in which
case the one-switch and shared past-hull policies coincide.

## Sources and reuse

Algorithm reference: [Marinov et al., Efficient Opportunistic Approachability](https://arxiv.org/abs/2602.21328).
Application/data precedent: [Liu and Garg, Redesigning Service Level Agreements](https://arxiv.org/abs/2410.14825).
The queueing and SLA experiment of Liu and Garg is a separate model; its
published results are not used as scores for this benchmark.

Code is available under the [MIT License](LICENSE). Public data retain their
source attribution and terms; see the data documentation. The original article
and third-party implementations are not included in this repository.
