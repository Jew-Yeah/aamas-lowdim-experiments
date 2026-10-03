[English](data_en.md) | [Русский](data_ru.md) | [Home](../README.md)

# Public data and finite demand profiles

We use the NYC Department of Parks and Recreation's public
[Forestry Service Requests](https://data.cityofnewyork.us/Environment/Forestry-Service-Requests/mu46-p9is)
dataset (`mu46-p9is`), selecting `SRCategory = Hazard`. The published
`CreatedDate` is the request-arrival field for this benchmark. We do **not**
use `InitiatedDate`: in the live release inspected during development it did
not represent the historical creation dates. We group the published floating
timestamps by their calendar date, without treating them as UTC timestamps.

## Access and provenance

Run `python scripts/download_nyc.py`. The anonymous public API performs
aggregation on the server. Only a calendar date, borough, and request count
are fetched. The cache holds one CSV per year and a JSON provenance record
with exact encoded queries, fetch time in UTC, response hashes, CSV hash,
excluded unknown-borough counts, and zero-day counts. Addresses, request IDs,
coordinates, names, and notes are never requested or stored. Repeated runs
verify and reuse the cache; `--refresh` intentionally takes a new snapshot.
The upstream dataset changes, so publish the checked aggregate snapshot
with the experiment. The downloader does not silently substitute generated data.

The default training year is 2019; the chronological holdout is 2021–2022
(730 daily rounds). We omit 2020 by design because the pandemic and severe
storm period create a separate stress-test question; this does not imply
that the holdout is stationary. Every calendar day is retained, including
zero-request days. Unknown boroughs are excluded and their totals recorded.
Duplicate reports of one incident remain separate registered requests. The
unit is a **request**, not a tree, inspection, completed service, or worker-hour.

## Adaptation to the article's game

Each day is a separate quota-planning round without a carried queue. The five demand coordinates
are request counts for Bronx, Brooklyn, Manhattan, Queens, and Staten Island.
They describe modeled service demand; we do not reconstruct a historical
inspection queue or infer the effects of changing actual staffing.

For the finite game we fit `M=6` demand centroids using only 2019 counts
(NumPy k-means++/Lloyd, fixed seed and eight starts). Raw Euclidean count
distance is used, without per-borough rescaling. Each held-out day is assigned
to its nearest centroid. The opponent observation is the corresponding
one-hot vertex of `Delta_M`, revealed after the allocation decision. The
payoff table uses the centroid demand, not the raw demand of that day.
Centroids can be fractional because they represent expected request counts.

This is an explicitly **quantized trace-derived game**. It does not establish
that the original trace has low affine dimension. The realized dimension
of its one-hot observations is the number of visited profiles minus one,
and is at most `M-1`. This cap is created by preprocessing. Record raw
counts, projected counts, daily L1/L2 residuals and their summaries alongside
finite-game results. Suggested sensitivity checks vary M, e.g. 3, 6 and 12.
Large storm-day residuals must remain visible rather than be clipped away.
`relative_l2_rms` is `sqrt(sum(error_l2**2)/sum(raw_l2_norm**2))`;
`relative_l1_total` is `sum(error_l1)/sum(raw_counts)`. These are reconstruction
errors, not learning errors. They are reported as null if the corresponding
raw-demand denominator is zero. For the initially fetched holdout and default
M=6, both ratios are substantial (approximately 37% and 34% respectively),
so raw-demand service diagnostics and profile-count sensitivity are essential.

Capacity, service thresholds, capacity cost and fairness scores are explicit
model parameters chosen before testing. The modeled shortfall is not a
counterfactual claim about NYC inspections. All compared methods receive
the same profiles, action set, capacity and feedback. Realistic queue/SLA
comparisons would need an additional stateful model and separate validation.

## Data-layer checks

`tests/test_data.py` checks server-side anonymous aggregation, complete
calendars, exclusion reporting, checksum verification, duplicate rejection,
empty-source failure, deterministic fitting, explicit quantization error,
simplex-vertex dimension and prevention of overlapping train/test years.
