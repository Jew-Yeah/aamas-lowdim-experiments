# Resource-service switching validation

[English](README.md) | [Русский](README.ru.md)

Complete fixed validation: **256 independent paired episodes per scenario, 1024 total**, all eight prespecified methods. The following are results of a constructed vector game, with inference conditional on its fixed stochastic generator.

## What the comparison establishes

- **One-switch − Fast: origin saddle**: Resolved lower own-target distance for one-switch. Outward 98.75% interval [-0.100547, -0.100473].

- **One-switch − Block safe only**: Both methods satisfy terminal membership on stored floats; no scientific ranking from KKT residuals. Outward 98.75% interval [-7.51699e-16, 2.89424e-13].

- **One-switch − Lag response**: Both methods satisfy terminal membership on stored floats; no scientific ranking from KKT residuals. Outward 98.75% interval [-1.06693e-13, 2.89424e-13].

- **One-switch − Window, W=16**: Both methods satisfy terminal membership on stored floats; no scientific ranking from KKT residuals. Outward 98.75% interval [-2.38482e-13, 2.89424e-13].

On the deterministic benign prefix, the exact average distance is 0.000117727 for one-switch and 0.177127 for safe only: difference -0.177009. This is a fixed-design computation, without a p-value or a claim of sampling superiority.

The recent-request fast saddle and the request-triggered balanced control are retained below. An improvement over origin-saddle fast does not establish that every valid fast oracle needs switching, or that one-switch is preferable to all causal controls. Physical shortage and activation costs can favor different policies.

## Fixed game and protocol

The horizon is T=262144, the origin-only prefix lasts H=98304 rounds, and the environment changes on H+1. The learner chooses p∈[0,1]. A request uses a genuine distinct resource coordinate from a catalogue fixed before play; an idle round is ℓ=0. There are M=T−H=163840 request coordinates.

```text
u(p, ℓ) = ((1−p) ε r, (1−p) ε ℓ, p(γ₀−(γ₀−γ₁)r)), r=Σℓᵢ
ε=1/√2, γ₀=1, γ₁=0.2, β=0.6386979044864147
p*(ℓ)=1 iff 2εr > γ₀−(γ₀−γ₁)r; equality chooses 0
S(Q)={0} before a request; after a request and origin:
S(Q)={(A,w,E): w≥0, Σwᵢ=A, E≥0, 2A+E≤β, support(w)⊆observed classes}
```

This is the full strict target, including benchmark responses at every interior hull mixture and tie-boundary limit. Endpoint distance is computed per episode before averaging. The analytical KKT projection retains its support gap and feasibility error; curves use certified full-target enclosures rather than an endpoint-only target surrogate. The fixed known Lipschitz bound is max(ε√(M+1),0.8√M).

The original budget **G=6T^(3/4)=69511.42501776238** and original block-safe base are used. The crossing round still uses fast; a fresh safe run starts on the next round. The master never periodically restarts. Original fast selects the observed origin in a tied saddle set; the recent-request control selects another valid tied saddle.

```text
After H: qₜ=0.05+0.9 sigmoid((θ−mean(previous p over memory))/0.03)
θ∼Uniform[0.45,0.55]; request iff shared Uₜ<qₜ, Uₜ∼Uniform[0,1]
```

Only past learner actions enter qₜ. Methods share θ and all exogenous uniforms, with a separate action history for each policy. Therefore reactive realized paths and own-hull targets can differ. Own-target distance contrasts measure adaptation to each policy's realized hull, not success against one identical service target. Exogenous q=0.5 and stationary-origin cells retain common realized paths. All policies use the same known game; θ, future uniforms, and evaluator outputs are unavailable to them.

The [fixed protocol](../../docs/resource_validation_protocol.json) and [public source commit](https://github.com/Jew-Yeah/aamas-lowdim-experiments/commit/dc404b3aeeebd73ba14c46e806b9d796cf59e899) are identified by the [precommitment receipt](precommitment.json), frozen at 2026-10-04T06:49:44.295022+00:00. The code and protocol were published before validation; the receipt was recorded locally before validation and published with these results. This is not registration in an external study registry. The design follows the earlier exploratory scalar studies retained in this repository. Development seeds 30000–30015 are excluded. Validation uses all seeds 40000–40255 in each cell; no seed replacement, optional stopping, scenario selection, or restriction to switched episodes is applied.

## All prespecified outcomes

Entries are episode means. Unserved is total unmet request units divided by **all T rounds**, not the fraction conditional on requests. Activation includes idle cost plus loaded overhead; idle is shown separately. These are modeled costs, without measured operating-energy or fairness claims. † means every stored episode satisfies analytical target membership; the parenthesis reports the largest numerical KKT displacement. Raw KKT means, quantiles, and all certificates remain in JSON.

### Reactive, memory 1 (primary)

| Method | Terminal distance | Unserved / round | Activation / round | Idle / round |
| --- | --- | --- | --- | --- |
| One-switch | 0† (KKT ≤ 2.93626e-13) | 0.342684 | 0.11229 | 0.0965333 |
| Fast: origin saddle | 0.10051 | 0.59377 | 3.8147e-06 | 3.8147e-06 |
| Block safe only | 0† (KKT ≤ 3.17801e-15) | 0.16193 | 0.226048 | 0.195949 |
| Lag response | 0† (KKT ≤ 1.07067e-13) | 0.156253 | 0.187498 | 0.156253 |
| Window, W=16 | 0† (KKT ≤ 2.42667e-13) | 0.233696 | 0.280516 | 0.268006 |
| Uniform, p=0.5 | 0.000702408 | 0.156558 | 0.374754 | 0.343442 |
| Request-triggered balanced control | 0† (KKT ≤ 3.44343e-15) | 0.0143043 | 0.378962 | 0.373906 |
| Fast: recent-request saddle | 0† (KKT ≤ 3.91742e-13) | 0.363285 | 0.029895 | 0.00622368 |

### Reactive, memory 16

| Method | Terminal distance | Unserved / round | Activation / round | Idle / round |
| --- | --- | --- | --- | --- |
| One-switch | 0† (KKT ≤ 2.9296e-13) | 0.34287 | 0.111912 | 0.0961969 |
| Fast: origin saddle | 0.100504 | 0.593761 | 3.8147e-06 | 3.8147e-06 |
| Block safe only | 0† (KKT ≤ 2.90046e-15) | 0.162296 | 0.225519 | 0.195519 |
| Lag response | 0† (KKT ≤ 2.33841e-14) | 0.105881 | 0.147058 | 0.10588 |
| Window, W=16 | 0† (KKT ≤ 9.18571e-14) | 0.146919 | 0.19717 | 0.165933 |
| Uniform, p=0.5 | 0.000520986 | 0.153013 | 0.377589 | 0.346987 |
| Request-triggered balanced control | 0† (KKT ≤ 3.48853e-15) | 0.0141106 | 0.379236 | 0.374248 |
| Fast: recent-request saddle | 0.0283816 | 0.475356 | 0.0299066 | 0.00623813 |

### Exogenous arrivals, q=0.5

| Method | Terminal distance | Unserved / round | Activation / round | Idle / round |
| --- | --- | --- | --- | --- |
| One-switch | 0† (KKT ≤ 2.91961e-13) | 0.275287 | 0.04464 | 0.0372105 |
| Fast: origin saddle | 0† (KKT ≤ 3.75533e-13) | 0.312435 | 3.8147e-06 | 3.8147e-06 |
| Block safe only | 0† (KKT ≤ 1.42941e-15) | 0.0591753 | 0.342239 | 0.291587 |
| Lag response | 0† (KKT ≤ 1.08288e-13) | 0.156276 | 0.187508 | 0.156276 |
| Window, W=16 | 0† (KKT ≤ 5.86892e-14) | 0.125662 | 0.224287 | 0.186933 |
| Uniform, p=0.5 | 0† (KKT ≤ 1.87766e-13) | 0.156217 | 0.375026 | 0.343783 |
| Request-triggered balanced control | 0† (KKT ≤ 9.05387e-14) | 0.112886 | 0.239544 | 0.199634 |
| Fast: recent-request saddle | 0† (KKT ≤ 2.69229e-13) | 0.250143 | 0.0747463 | 0.0622879 |

### Stationary origin control

| Method | Terminal distance | Unserved / round | Activation / round | Idle / round |
| --- | --- | --- | --- | --- |
| One-switch | 3.8147e-06 | 0 | 3.8147e-06 | 3.8147e-06 |
| Fast: origin saddle | 3.8147e-06 | 0 | 3.8147e-06 | 3.8147e-06 |
| Block safe only | 0.0627217 | 0 | 0.0627217 | 0.0627217 |
| Lag response | 1.90735e-06 | 0 | 1.90735e-06 | 1.90735e-06 |
| Window, W=16 | 1.90735e-06 | 0 | 1.90735e-06 | 1.90735e-06 |
| Uniform, p=0.5 | 0.5 | 0 | 0.5 | 0.5 |
| Request-triggered balanced control | 1.90735e-06 | 0 | 1.90735e-06 | 1.90735e-06 |
| Fast: recent-request saddle | 3.8147e-06 | 0 | 3.8147e-06 | 3.8147e-06 |

## Primary paired inference

Differences are **one-switch minus comparator**; negative favors one-switch. The independent complete episode is the sampling unit. The table retains numerical t/Holm/bootstrap diagnostics for all four planned tests, and separately gives outward intervals incorporating endpoint numerical uncertainty. Ordinary intervals use 95%; the four-comparison family uses 98.75% marginal intervals. Tiny differences between two membership-zero methods are floating-summation effects and have no scientific direction, even when a raw numerical p-value is small.

| Comparator | Raw numerical mean Δ | Outward 95% CI | Outward 98.75% CI | Bootstrap 95% | Bootstrap 98.75% |
| --- | --- | --- | --- | --- | --- |
| Fast: origin saddle | -0.10051 | [-0.100539, -0.100481] | [-0.100547, -0.100473] | [-0.10054, -0.100481] | [-0.100548, -0.100474] |
| Block safe only | 2.88479e-13 | [-7.34979e-16, 2.89365e-13] | [-7.51699e-16, 2.89424e-13] | [2.88266e-13, 2.8869e-13] | [2.88208e-13, 2.88747e-13] |
| Lag response | 1.82484e-13 | [-1.06688e-13, 2.89365e-13] | [-1.06693e-13, 2.89424e-13] | [1.82274e-13, 1.8269e-13] | [1.82214e-13, 1.82742e-13] |
| Window, W=16 | 5.08807e-14 | [-2.38437e-13, 2.89365e-13] | [-2.38482e-13, 2.89424e-13] | [5.06242e-14, 5.11297e-14] | [5.05584e-14, 5.11973e-14] |

| Comparator | Raw p diagnostic | Holm diagnostic | Status | Raw W/T/L | KKT W/U/L | Both membership-zero |
| --- | --- | --- | --- | --- | --- | --- |
| Fast: origin saddle | underflow; log p=-1547.49 | log10 p=-671.465 | ok | 256/0/0 | 256/0/0 | 0 |
| Block safe only | underflow; log p=-1303.08 | log10 p=-565.443 | ok | 0/0/256 | 0/256/0 | 256 |
| Lag response | underflow; log p=-1193.19 | log10 p=-517.896 | ok | 0/0/256 | 0/256/0 | 256 |
| Window, W=16 | underflow; log p=-819.084 | log10 p=-355.724 | ok | 0/0/256 | 0/256/0 | 256 |

W/T/L are numerical wins/exact-floating ties/losses; W/U/L use KKT endpoint bounds with unresolved comparisons. Both-membership-zero counts use the separate analytical diagnostic. P-values that underflow are retained on a finite log scale; a displayed underflow is not mathematical p=0. Constant differences have undefined tests. Bootstrap uses 9999 paired whole-episode draws with fixed seed 2026100403 and percentile intervals. Its raw intervals do not account for oracle uncertainty; compare the outward intervals. The remaining scenarios and uniform/trigger/recent-saddle controls are descriptive.

## Endpoint distribution and numerical resolution

Two separate checks are retained. **KKT Z/P/U** uses upper=0 / lower>0 / unresolved numerical distance. **Membership** evaluates the proven full-target inequality on stored aggregate/activation totals. Within the model A=Σw, yet summing these two forms in different orders can leave a nonzero KKT displacement. Membership-zero does not mean a formal exact-real-arithmetic certificate. No arbitrary cutoff chooses which algorithm wins. CP denotes exact two-sided 95% Clopper–Pearson intervals; their sampling exactness does not validate floating arithmetic.

### Reactive, memory 1 (primary)

| Method | KKT Z/P/U | CP definite-positive | CP possible-positive enclosure | Membership zero | CP outside-membership | Min β−2A−E |
| --- | --- | --- | --- | --- | --- | --- |
| One-switch | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.039219 |
| Fast: origin saddle | 0/256/0 | [0.985694, 1] | [0.985694, 1] | 0/256 | [0.985694, 1] | -0.202204 |
| Block safe only | 1/0/255 | [0, 0.0143064] | [0, 0.999901] | 256/256 | [0, 0.0143064] | 0.182531 |
| Lag response | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.2297 |
| Window, W=16 | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.0216726 |
| Uniform, p=0.5 | 0/51/205 | [0.152077, 0.253472] | [0.152077, 1] | 205/256 | [0.152077, 0.253472] | -0.0161313 |
| Request-triggered balanced control | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.239507 |
| Fast: recent-request saddle | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.0941358 |

### Reactive, memory 16

| Method | KKT Z/P/U | CP definite-positive | CP possible-positive enclosure | Membership zero | CP outside-membership | Min β−2A−E |
| --- | --- | --- | --- | --- | --- | --- |
| One-switch | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.039185 |
| Fast: origin saddle | 0/256/0 | [0.985694, 1] | [0.985694, 1] | 0/256 | [0.985694, 1] | -0.202295 |
| Block safe only | 3/0/253 | [0, 0.0143064] | [0, 0.997577] | 256/256 | [0, 0.0143064] | 0.18268 |
| Lag response | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.335567 |
| Window, W=16 | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.230746 |
| Uniform, p=0.5 | 0/47/209 | [0.138128, 0.236552] | [0.138128, 1] | 209/256 | [0.138128, 0.236552] | -0.0159087 |
| Request-triggered balanced control | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.239507 |
| Fast: recent-request saddle | 0/256/0 | [0.985694, 1] | [0.985694, 1] | 0/256 | [0.985694, 1] | -0.0645228 |

### Exogenous arrivals, q=0.5

| Method | KKT Z/P/U | CP definite-positive | CP possible-positive enclosure | Membership zero | CP outside-membership | Min β−2A−E |
| --- | --- | --- | --- | --- | --- | --- |
| One-switch | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.201797 |
| Fast: origin saddle | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.193979 |
| Block safe only | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.210977 |
| Lag response | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.228003 |
| Window, W=16 | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.234832 |
| Uniform, p=0.5 | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.0421249 |
| Request-triggered balanced control | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.239507 |
| Fast: recent-request saddle | 0/0/256 | [0, 0.0143064] | [0, 1] | 256/256 | [0, 0.0143064] | 0.207162 |

### Stationary origin control

| Method | KKT Z/P/U | CP definite-positive | CP possible-positive enclosure | Membership zero | CP outside-membership | Min β−2A−E |
| --- | --- | --- | --- | --- | --- | --- |
| One-switch | 0/256/0 | [0.985694, 1] | [0.985694, 1] | 0/256 | [0.985694, 1] | — |
| Fast: origin saddle | 0/256/0 | [0.985694, 1] | [0.985694, 1] | 0/256 | [0.985694, 1] | — |
| Block safe only | 0/256/0 | [0.985694, 1] | [0.985694, 1] | 0/256 | [0.985694, 1] | — |
| Lag response | 0/256/0 | [0.985694, 1] | [0.985694, 1] | 0/256 | [0.985694, 1] | — |
| Window, W=16 | 0/256/0 | [0.985694, 1] | [0.985694, 1] | 0/256 | [0.985694, 1] | — |
| Uniform, p=0.5 | 0/256/0 | [0.985694, 1] | [0.985694, 1] | 0/256 | [0.985694, 1] | — |
| Request-triggered balanced control | 0/256/0 | [0.985694, 1] | [0.985694, 1] | 0/256 | [0.985694, 1] | — |
| Fast: recent-request saddle | 0/256/0 | [0.985694, 1] | [0.985694, 1] | 0/256 | [0.985694, 1] | — |

If every endpoint is unresolved under KKT bounds, the possible-positive population enclosure may be [0,1]; the independent analytical membership check supplies more information in this game. Membership margins are shown only for episodes with requests. No observed outside-membership episodes does not prove equality or practical equivalence in the generator population. A 0/256 count still has a positive upper CP limit. The JSON also retains combined stored-float zero/positive/unresolved counts and all residual maxima.

## Budget crossing and conditional safe tails

| Scenario | Switched / all | Exact 95% fraction CI | τ 5/50/95% (switched only) | Mean safe rounds (all) | Tail bound check |
| --- | --- | --- | --- | --- | --- |
| Reactive, memory 1 (primary) | 256/256 | [0.985694, 1] | 171376/171472/171576 | 90672.5 | pass |
| Reactive, memory 16 | 256/256 | [0.985694, 1] | 171379/171476/171570 | 90670.2 | pass |
| Exogenous arrivals, q=0.5 | 256/256 | [0.985694, 1] | 236729/237342/237988 | 24802.2 | pass |
| Stationary origin control | 0/256 | [0, 0.0143064] | — | 0 | pass |

The probability denominator retains every episode. Crossing quantiles are conditional descriptions, not a filtered performance comparison. Safe-tail checks use the tail's own observed hull and h·δ_tail≤6h^(3/4) with the saved numerical tolerance 2e−7. They check the original instantiated bound computationally; the data do not replace its mathematical proof. Master's E freezes after crossing; fast-only E belongs to a different counterfactual closed-loop path and is labeled separately.

## Figures

- [Switching dynamics and paired distance differences](figures/resource_switching_dynamics.pdf): fixed phase boundary, master/fast residuals relative to the budget, distance enclosures and paired differences.
- [Endpoint outcomes and scenario sensitivity](figures/resource_outcomes_and_sensitivity.pdf): all controls, terminal distance, physical unmet demand, idle/loaded activation cost and crossing probability.

The English [captions](figures/captions.json) identify every quantity. Dark curve fill is numerical target enclosure; the outer light fill additionally includes 95% pointwise whole-episode bootstrap uncertainty (2000 draws, seed 2026100404). Neither is a simultaneous confidence band. A curve's midpoint is a displayed enclosure midpoint, not an exact distance observation. The fixed full-trace seeds are 40000 and 40001.

The first request expands the target from {0} to the full resource polytope. A sharp distance drop near H+1 can therefore reflect target expansion, without immediate physical learning or elimination of unmet demand. Zero target distance is compatible with positive physical shortage; inspect both the distance and shortage/cost panels.

## Use in the paper

Use this as a synthetic mechanism test of the specified one-switch instantiation. The main figure is the switching-dynamics figure; keep all eight controls and all four scenarios in a full table in the appendix or supplementary material, with the strong controls disclosed in the main discussion. Discuss the benign-prefix benefit separately from terminal fallback, and disclose target expansion and policy-dependent realized hulls. The following English paragraph is generated from the completed results. It does not alter the manuscript or assert year-specific conference formatting requirements.

> We evaluated a fixed synthetic vector resource-service game using 256 independent paired episodes in each of four prespecified scenarios (eight causal policies; T=262144, H=98304, G=6T^{3/4}). The primary reactive environment used only past service activation. One-switch satisfied analytical terminal membership on stored floats in 256/256 episodes, whereas fast with the specified origin saddle tie had mean full-target distance 0.10051. The numerical paired mean difference was -0.10051, with an outward 98.75% interval [-0.100547, -0.100473] accounting for endpoint numerical uncertainty. On the deterministic origin-only prefix, mean distance was 0.000117727 for one-switch and 0.177127 for block safe; this comparison has no sampling p-value. An alternative valid fast saddle choice and the request-triggered balanced control also satisfied terminal membership in 256/256 and 256/256 episodes, respectively, limiting any claim that switching is generally necessary. Reactive policies may induce different paths and own-hull targets; zero target distance can coexist with unmet demand, and the first request itself expands the target. We therefore report all controls and separate physical shortage and activation costs.

## Reproduction and limits

From the repository root, use this result release, whose critical simulator and protocol match the frozen source commit, and a fresh output directory; the runner refuses to overwrite completed episodes. The pilot is optional for reproduction and does not contribute to inference.

```sh
python -m pip install -r requirements-lock.txt
python -m pip install -e ".[test]"
python -m pytest tests/test_resource_validation.py tests/test_resource_target.py
python scripts/run_resource_validation.py --stage pilot --output results/resource_validation_replay
python scripts/run_resource_validation.py --stage freeze --output results/resource_validation_replay
python scripts/run_resource_validation.py --stage validation --output results/resource_validation_replay
python scripts/run_resource_validation.py --stage analyze --output results/resource_validation_replay
python scripts/build_resource_validation_figures.py --output results/resource_validation_replay
python scripts/build_resource_validation_report.py --output results/resource_validation_replay
```

The report and figure builders are postprocessors supplied with this result release; the frozen receipt covers the protocol, simulator, target oracle, paired-inference code, and original learners. [Statistical summary](analysis.json), [distribution diagnostics](distribution_diagnostics.json), and [SHA-256 manifest](reproducibility_manifest.json) retain exact filenames and hashes. Episode tables reside in `primary/episodes.json`, `memory16/episodes.json`, `exogenous/episodes.json`, and `stationary/episodes.json`. NPZ dynamics and representative raw arrays are kept locally as large generated artifacts; regenerate them with the fixed seeds and verify their saved SHA-256 values rather than assuming that a public checkout contains them. Frozen runtime versions: python 3.12.14, numpy 2.5.3, scipy 1.18.1.

These findings concern one finite synthetic game and one specified nonanticipating reaction rule. They do not establish representativeness for real resource systems, attacker learning, empirical minimax rates, or a q=4 result. Changing T would change the catalogue dimension and Lipschitz constant. All controls and sensitivity cells remain reportable, including zero-distance ties and negative comparisons; there is no post hoc selection of a winning method or data regime.
