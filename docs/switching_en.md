[English](switching_en.md) | [Русский](switching_ru.md) | [Home](../README.md)

# Full one-switch trajectories

The earlier three-mode CAGE study cannot activate the original certified switch:
with exact projections, only two new pure modes can add residual, so E is at most
4. Increasing the number of seeds or the horizon cannot fix this limitation.
The new studies therefore separate a real request-share tracking game from a
constructed test of the original block-safe budget. The core master, fast policy,
and block-safe implementation remain unchanged.

## Chronological NYC resource tracking

The source is the official [Forestry Service Requests dataset](https://data.cityofnewyork.us/Environment/Forestry-Service-Requests/mu46-p9is/data).
We use cached daily Hazard request counts for 2021 and 2022, covering 730 days.
The aggregate CSVs exactly match the hashes retained from their original retrieval.
They contain counts only. These are registered requests, not staffing requirements,
unserved queues, or observed outcomes of our allocations.

The three modeled resource pools are Brooklyn, Queens, and the other three
boroughs. Each observed demand share is the corresponding daily count divided
by the total. One zero-count day is represented by uniform shares. There is no
profile fitting, quantization, or learned calibration in this tracking study.

The known game is P=L=Delta_3, u(p,ell)=(p-ell)/sqrt(2), with benchmark response
p*(ell)=ell. Every benchmark payoff is zero, so the **entire strict target is
{0}**, including responses to unobserved hull mixtures.

The master uses a game-specific lag safe base: start uniformly on its fresh
first round, then allocate the previous observed share. On every h-round path,
the cumulative payoff telescopes to (p_initial-ell_last)/sqrt(2). Simplex
diameter therefore gives the conservative certificate B0(0)=0 and B0(h)=1
for h>0. Hence G=1 is known before observations. It is not a fitted budget,
a replacement constant in the original block routine, or a claim that this is
the tight uniform-start bound.

All comparators choose before observing the current share and start uniformly.
Window uses the previous at most 16 shares. No learner restarts within a run.
The two annual checks are separate fresh runs, not subdivisions of the primary
730-day trajectory. No Hedge on the coordinate sum is reported, because that
sum is identically zero in this balance game.

## Original-budget mechanism diagnostic

The second game has capacity p in [0,1], opponent set conv{0,e_1,...,e_M} in
Euclidean R^M, demand r(ell)=sum(ell), payoff u=p-r, and benchmark p*=r.
Again the full target is {0}. The first 128 rounds have zero demand; every later
round introduces a new orthogonal type with unit demand. T=16384 is announced.

This run retains the original block-safe routine and **G=6 T^(3/4)**. Closed-form
fast saddle and hull projection operations reproduce the original equations.
In particular a new e_i projects onto zero. Using a standard simplex embedding
or compressing these coordinates before fast projection would change the game.
The safe routine alone uses the exact payoff compression (1-r,r), which preserves
its gradients, block averages, response payoffs, and all updates.

Novel types are payoff-equivalent. This deliberately pathological test exposes
the fast algorithm's sensitivity to the given geometry; it is not realistic
attacker learning, a low-dimensional rate experiment, or a q=4 validation.
Standalone block-safe and a certified lag control are included and outperform
the master. The defensible comparison is the benefit relative to leaving the
fast policy uninterrupted.

## Evaluation and reproduction

We distinguish delta_t=norm(mean vector payoff) from the mean daily norm or
absolute imbalance. Positive and negative errors can cancel in delta_t. Each
plot of cumulative differences explicitly identifies the comparator and metric;
negative differences favor one-switch. Residual monitoring stops after crossing;
the dashed fast-only trace is a separately run counterfactual, not continuing
master monitoring. The crossing action stays fast and the fresh safe run starts
the next round.

These scenario additions are exploratory and were motivated by the review of
the non-switching CAGE study. The fixed chronological traces do not carry iid-day
bootstrap intervals, p-values, or confirmatory holdout claims. The log-distance
NYC panel has a disclosed 1e-5 display floor; raw arrays retain all values.
Numerical fast certificates are floating-point checks, not exact arithmetic proofs.

```powershell
python -m pytest -q
python scripts/run_switching_study.py
```

The script verifies source CSV hashes, replays the 730-day trace and both years,
runs the original-budget diagnostic, and exports English PNG/PDF dynamics.
[Results and all controls](../results/switching/README.md) include source hashes,
software versions, per-step NPZ arrays, analytical target definitions, and the
JSON protocol. No original CAGE inputs or scientific results are overwritten.
