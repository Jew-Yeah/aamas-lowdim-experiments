[English](README.md) | [Русский](README.ru.md)

# AAMAS experimental materials

The current presentation centers on the full chronological NYC switching trace.
[experiments.tex](experiments.tex) is the insertable main section;
[supplementary_methods.tex](supplementary_methods.tex) contains detailed methods
and additional results. The full manuscript remains outside this repository;
theory statements and proofs are preserved in the current integration.

Two instantiations of the master are distinguished:

- NYC uses a separately certified lag safe base and conservative **G=1**.
  Crossing is round 44 of 730, followed by the complete 686-round safe tail.
- The constructed resource-balance test retains the **original block-safe
  routine and G=6 T^(3/4)**. At T=16384 it crosses at 8817. Its high-dimensional,
  payoff-equivalent labels test the mechanism, not low-dimensional rate exponents.

All five methods are retained: one-switch, fast-only, lag-safe, Window, and
standalone block-safe. Final vector distance and average daily mismatch differ;
signed errors can cancel in the former. The figures show benefit against fast
and positive differences against stronger controls. These fixed post-review
exploratory traces carry no confidence bands or significance claims.

The earlier CAGE comparison is frozen and secondary: lower scalar loss than
tuned Hedge, higher than Window, and no switch. Its original observations,
selection, statistical intervals and figures remain unchanged.

## Figures and integration

Current English figures are mirrored here:

- [NYC dynamics PDF](figures/switching_nyc_dynamics.pdf) and
  [PNG](figures/switching_nyc_dynamics.png).
- [Original-budget diagnostic PDF](figures/switching_original_budget.pdf) and
  [PNG](figures/switching_original_budget.png).

Copy PDFs into the manuscript's `figures/` directory and use
`\input{experiments}`. Keep the official AAMAS class, one shared bibliography,
figure captions below, table captions above, and accessibility descriptions.
[experiment_references.bib](experiment_references.bib) supplies source entries
to merge into that bibliography. The compiled whole manuscript determines
page compliance; the current combined page count is unverified. Local compilation
of the updated project failed because the native compiler runtime is unavailable.

The NYC log-distance panel has a disclosed `1e-5` display floor; raw arrays
retain lower values. Differences use **master minus comparator** and explicitly
name the daily norm or absolute-imbalance metric. The crossing action stays fast;
safe starts next round. Old CAGE intervals retain their original, separate scope.

## Reproduction and reviewer materials

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/run_switching_study.py
```

The new studies require only base dependencies. The recorded development suite
passed **152 tests**. [Full results](../results/switching/README.md),
[method notes](../docs/switching_en.md), and
[JSON protocol](../results/switching/protocol.json) retain settings, hashes,
analytical targets, all controls and per-round arrays.

For blind review, cite the current anonymous supplement rather than the
identifying public repository. Include the new switching modules, aggregate NYC
inputs, runner, results and figures; an earlier CAGE-only package is insufficient.
Keep theory/Lean proofs and current experimental methods as separate, identifiable
components of the single package. Read [submission guidance](../docs/aamas_submission_en.md)
and [AI assistance](ai_assistance.md). Preserve author direction, technical
assistance, and the stated limits of the retained historical record.
