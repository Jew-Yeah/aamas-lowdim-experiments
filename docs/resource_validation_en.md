[English](resource_validation_en.md) | [Русский](resource_validation_ru.md) | [Home](../README.md)

# Full vector resource-allocation validation

This study checks early approachability and later fallback in a constructed resource model. It retains aggregate unmet demand, every request class's unmet demand, and resource activation cost as separate payoff coordinates. A benefit over uninterrupted fast-only is a specific claim; it does not imply superiority over safe-only or the strong controls. Results are not yet asserted by these method notes.

The [design protocol](resource_validation_protocol.json) follows the earlier scalar studies and their mixed results. It requires an auditable source/protocol receipt before the untouched validation seeds are run. Publication of fixed code and protocol is planned before validation where available; the receipt distinguishes a local commit from an actually published commit. This is local precommitment, not an external preregistration. Parameters were chosen from the stated mathematical budget condition before development outputs; no favorable method ranking is used to select the horizon.

## Game and target

The learner chooses an activation fraction p in [0,1]. An opponent action ell lies in conv{0,e_1,...,e_M}; origin is an idle round and a unit vector requests one unit from a particular class. Each postchange round has its own potential new class, so different requests have genuinely different payoff coordinates. The catalogue and payoff map are fixed; adaptive selection does not alter an existing class.

Let r=sum(ell) and epsilon=1/sqrt(2). The normalized vector payoff is

`u(p,ell)=((1-p)*epsilon*r, (1-p)*epsilon*ell, p*(1-0.8*r))`.

Its coordinates describe aggregate shortage, class-resolved shortages, and modeled activation cost. Idle activation costs one unit, loaded activation 0.2 units. Vertex norms are at most one. These are modeled quantities, not measured service outcomes or operating costs. Class-resolved constraints give visibility into service distribution; no empirical fairness improvement is presumed.

The finite game has M=T−H=163840 potential classes. A known Euclidean Lipschitz bound is `K_u=max(epsilon*sqrt(M+1), 0.8*sqrt(M))≈323.817`, fixed before play. This horizon uses one fixed game valid for every opponent path. Changing T would change the game dimension and this constant; the experiment does not estimate low-dimensional convergence rates.

The fixed all-coordinate weighted benchmark chooses on when `2*epsilon*r > 1-0.8*r`, and off otherwise. Equality chooses off. Put `beta=2*epsilon/(2*epsilon+0.8)=0.6386979044864147`. Before any request, the full strict target is {0}. After a request, with the observed origin included, a target point (A,w,E) satisfies

`w>=0, sum(w)=A, E>=0, 2*A+E<=beta`,

where w is supported on the observed request classes. This is the full closed convex response target, including all interior opponent mixtures and limiting response cells. Evaluating only observed benchmark response points would miss this target.

Primary terminal distances use this complete target with an independent analytical/support-gap check. All-round dynamics retain certified lower and upper bounds; exact distances at fixed checkpoints are used where feasible. Numerical envelopes are labeled separately from uncertainty due to finite episode count.

## Four fixed scenarios and eight methods

T=262144 and H=98304=3T/8. The first H rounds are origin only, a deliberately benign constructed phase. The original k=1 block safe base and budget remain `G=6*T^(3/4)=69511.42501776238`. The pre-study budget-margin condition is `G/T <= beta*(H/T)/(2*epsilon-beta)`: 0.265165 <= 0.308842. This condition is not a guarantee of superiority or switching in every random episode.

| Scenario | Postchange opponent |
|---|---|
| Primary | Reactive, memory one past action |
| Memory 16 | Same reaction using 16 past actions |
| Exogenous | Request probability 0.5, independent of learner actions |
| Stationary | Origin only for the entire horizon |

For a reactive episode, theta is sampled uniformly from [0.45,0.55]. Given the mean of already observed learner actions, request probability is

`q_t=0.05+0.9*sigmoid((theta-mean_past_p)/0.03)`.

A shared uniform U_t determines whether the current potential class requests service. Current p_t is unavailable to this attacker policy. Every method receives the same theta and indexed exogenous innovations within an episode, while its attacker uses that method's own past action history. Consequently interactive paths and realized class sets can differ. This comparison measures performance against the same opponent policy, not replay against an identical realized trajectory.

The methods are one-switch, fast-only, original block-safe-only, response to the previous observation, benchmark response to the previous at most 16 observations, constant uniform activation, and a request-triggered control. All start at p=0.5. The trigger is off after initialization until the first observed request, then uses p=beta forever. It is an intentionally strong, causal, game-specific control; no certified-base status is asserted here. It must remain in the report even if it performs best. The switch-crossing action is still fast; a fresh safe run begins next round, without periodic restarts.

An eighth mandatory control, `fast_recent_saddle`, keeps the uninterrupted fast policy and original hull geometry but uses another valid dual tie rule: the most recently observed request replaces the origin in the relevant tied saddle set after the first request. This mathematical alternative was specified before development outputs. The primary comparison is conditional on the chosen origin-tie oracle; a benefit over it does not prove that every valid fast oracle needs fallback. The alternate control remains descriptive and must be retained even if it performs better.

An important limitation is that interactive methods can have different Q and therefore different S(Q). Primary distances describe each method's own full strict target; their differences are not success against one identical service goal. Physical shortage and idle-cost comparisons accompany them. A common full-library/aggregate target diagnostic, if shown, is secondary and can be weaker than the realized-hull target.

## Independent validation and inference

Development uses seeds 30000–30015. Validation uses all seeds 40000–40255: 256 complete episodes per scenario, 1024 in total. Selected full paths use seeds 40000 and 40001, fixed before validation. All cases and methods are retained, including no switches. Neither rounds nor payoff coordinates count as independent replicates.

The origin-only prechange path is deterministic for every method. Its exact every-round mean target distance and one-switch-minus-safe difference are reported as fixed-design quantities. Repeating them under many seeds does not justify a stochastic t-test or a statistical superiority claim. No prefix noise is added to manufacture such a test.

Four primary comparisons in the memory-one scenario use terminal full-target distance: one-switch minus fast, safe, lag, and window. Negative favors one-switch. The predictions are early speed and improvement over continued fast; the other terminal comparisons are uncertain. Use two-sided paired t tests, Holm across these four tests, ordinary 95% mean-difference intervals and Bonferroni 98.75% marginal intervals. [NIST describes paired-difference inference](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm).

With n=256 and conservative alpha=0.05/4, the normal-approximate 80% detection threshold is paired d_z≈0.2087; noncentral-t power at d_z=0.22 is about 84% under independent normal differences. This is a planning assumption, not an observed-power or practical-effect guarantee. Exact target membership can produce many zero distances, so actual differences need not be normal; zero fractions and paired win/tie/loss counts accompany the mean results and bootstrap robustness checks. [NIST gives the sample-size planning formula](https://www.itl.nist.gov/div898/handbook/prc/section2/prc222.htm).

Robustness intervals use 9999 resamples of whole paired episodes. Mean curves use 2000 whole-episode draws with shared weights at all times, yielding pointwise 95% intervals, not simultaneous bands or individual-path ranges. [SciPy documents paired bootstrap resampling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Bounds on numerical distance and sampling intervals must remain distinguishable. For approximate endpoint results, a paired true difference lies between lower(master)−upper(control) and upper(master)−lower(control).

Report exact episode-level distance before averaging distances, all physical coordinates, switch probability and its binomial interval, crossing delays and safe-tail lengths. A distance computed from the average vector across episodes is a different quantity. Constant differences have undefined t statistics and explicit degenerate status; very small p-values require finite log-scale reporting rather than mathematical p=0. All-zero observed distances mean no failures in the sample, not population equality or established equivalence; include binomial uncertainty for positive-distance frequency and separate numerically unresolved cases. Sensitivity scenarios and trigger/uniform comparisons are descriptive. Scientific failures, losses and oracle defects are recorded; no replacement seeds or selected winning cases.

The code, receipt, exact endpoint certificates, all episode summaries and dynamics bounds make the study reproducible. Inference applies to the stated stochastic model only. The large dimension, benign prefix, changing interactive targets and specialized trigger are material limitations. Article changes wait for validated results.
