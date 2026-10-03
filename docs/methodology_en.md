[English](methodology_en.md) | [Русский](methodology_ru.md) | [Home](../README.md)

# Experimental protocol

## Common game

There are eight service schedules and a fixed library of M demand profiles.
The learner chooses a mixture of schedules; the environment reveals a mixture
of profiles. A fixed tensor contains five unmet-demand coordinates, one resource
cost, and one disparity coordinate. Mixture payoffs are biaffine.
Each mixture averages the losses of individual schedules and profiles. Service
metrics are expected outcomes under the schedule distribution, including the
expected disparity of the selected schedule.

Each schedule's deficit is the positive part of demand minus its quota.
The baseline capacity C is 0.6, 0.8, or 1.0 times the median total training demand.
Schedules include equal quotas, historical demand shares, five focused plans,
and a reserve plan with 1.25C total capacity. Thus C denotes baseline capacity;
the reserve option is an explicit permitted action. Quotas are continuous
service-capacity units, allowing fractional expected request counts.

Resource cost is 0.25 times total planned capacity divided by C. It is a
dimensionless model parameter, not observed financial expenditure. Disparity is
the maximum minus minimum unmet demand across boroughs. All deficits and
disparity are divided by C. The tensor is then divided by the maximum vertex
payoff norm, or by one if this norm is smaller. This normalization is fixed
before play and gives ||u||_2 <= 1.

The benchmark response minimizes a fixed weighted scalar loss, with weights
(1,1,1,1,1,1,0.5), and selects the smallest action index in a tie. No robust
threshold feasibility claim is attached to this initial benchmark.

## Compared policies

All methods receive the same tensor, response map, announced horizon, and past
observations. A method chooses p_t before receiving ell_t. Uniform and historical
share policies select fixed permitted schedules. The previous-week policy selects
the response to the mean of up to seven past profile mixtures.
The fixed-reserve reference always selects the permitted 1.25C schedule. It
helps distinguish the value of adaptation from the benefit of extra capacity.

The past-hull method is implemented once and labeled as shared with the method
of Marinov et al. The one-switch policy adds the manuscript's threshold and a
fresh block-safe tail. The safe routine is also run independently. The
exponential-cover algorithm of Marinov et al. Theorem 21 is not implemented.

With eight schedules the learner's affine dimension is seven. The default
budget is G_T=6 sqrt(7) T^(3/4). Since each residual is at most two, no switch is
possible for the 730-day NYC run. Its one-switch and past-hull actions should
coincide. Unit tests exercise threshold crossing in a valid abstract-base game.

## Measurements

The primary mathematical measurement is the Euclidean distance of the average
normalized payoff to S(Q_t), where Q_t is the full hull of revealed actions.
Response-cell polyhedra and independent support LPs include responses at
unobserved mixtures. A squared primal–dual gap supplies an additive distance
error bound sqrt(gap). The implementation reports numerical floating-point
diagnostics, not a formal exact-arithmetic proof.

NYC service measurements use the original held-out daily counts: expected
unmet requests, borough deficits, service fractions, resource cost, and deficit
disparity. The learner still chooses before seeing that day's demand. These
measurements describe the stated allocation model; historical inspections,
queues, routing, and counterfactual SLA waiting times are not simulated.

## Chronology and sensitivity

Profile fitting and schedule construction use only 2019. The chronological test
sequence is 2021 followed by 2022, including zero-demand days. The finite game
assigns each day to its nearest fixed training profile. Results include M=3,6,12
and three baseline-capacity ratios. Quantization error is reported independently.
The realized dimension is the number of visited one-hot profiles minus one,
and is at most M-1. This dimension refers to profile coordinates; raw borough
counts and the number of boroughs have separate meanings.

Controlled synthetic paths introduce q+1 unknown regimes in blocks and then mix
them, for q=1,...,5 and three recorded seeds. They measure adaptation to new
regimes; they do not estimate worst-case convergence exponents. A separate
layered-paraboloid experiment checks the manuscript's geometric moment bound
at finite horizons. It does not establish a game minimax lower bound.

## Reproduction

Every run records its configuration, tensor hash, seeds, software versions,
actions, numerical residuals, and measured outcomes. The reference report links
the exact commands and committed outputs. Wall-clock measurements include
post-hoc evaluation and are descriptive; they are not a per-round complexity
comparison across hardware. Automated tests verify causality, saddle residuals,
projections, tie rules, full-target construction, safe updates, and preprocessing.
