[English](aamas_submission_en.md) | [Русский](aamas_submission_ru.md) | [Paper files](../paper/README.md)

# AAMAS 2027: experimental presentation and submission

Checked on 4 October 2026 against the official main-track instructions, Q&A,
and formatting kit. The retained manuscript is an AAMAS 2027 draft.
Requirements below come from the conference; figure selection and statistical
presentation recommendations are our decisions for this particular study.

## Official requirements

The [submission instructions](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/instructions/)
require an English anonymous PDF made with LaTeX and the unchanged official
template. Main-track content has a maximum of eight pages; additional pages
are for bibliography. Essential setup, results and evaluation information
belong in the main paper. Supplement is optional, anonymous, a single ZIP,
and at most 25 MB; reviewers need not consult it. A public archival release
and reference belong in the camera-ready publication.

The [official Q&A](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/qa/)
clarifies that an appendix inside the main PDF counts toward eight pages.
For AI-assisted methodology, report the tool, available relevant prompts/history
and version information; disclose missing records instead of reconstructing
them. The [AI assistance statement](../paper/ai_assistance.md) records the
experimental contribution and retained prompts, with explicit limitations.

The [formatting kit](https://warwick.ac.uk/fac/sci/dcs/aamas2027/aamas_2027_template.zip)
uses `\documentclass[sigconf,anonymous]{aamas}`. It places figure captions
below figures and table captions above tables, and requires a plain-text
`\Description` of up to 2000 characters for each substantive figure. Graphs
must remain intelligible in grayscale and with color-vision deficiencies.
Use embedded fonts and inspect figures at their final column width. Our
English PDF exports use explicit labels, distinct line styles/markers and
unchanged numerical data; no document layout parameters are modified.

The [main-track call](https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/call-for-main-track/)
includes soundness, reproducibility and clarity among review criteria. The
checked documents do not prescribe a particular seed count, number of
baselines, confidence-interval method, or mandatory separate experimental
checklist. Our splits and paired bootstrap are methodological choices, not
rules asserted on behalf of AAMAS.

## Recommended main-paper contents

Use [experiments.tex](../paper/experiments.tex), comprising setting and scope,
methods and validation, metrics and uncertainty, a primary table, a two-panel
cumulative difference figure, and interpretation/limitations. The table gives
the three selected means and both original simultaneous primary intervals.
The figure shows when each paired difference accumulates across attacker
phases. It uses descriptive pointwise bands, explicitly distinguished from
the primary intervals.

This is the compact default for a theory paper with an applied illustration.
Add `calibrated_geometry.pdf` only if space remains; preserve the text's
distinction between native held-out cost and calibrated vector distance.
The original latest manuscript fills eight content pages. Inserting the
experimental section requires a separate allocation of space in the full
paper; its final page count must be checked after integration.

For supplementary material and GitHub, use the English
[publication figure pack](../results/cage_adaptation/aamas/README.md), the
[expanded methods](../paper/supplementary_methods.tex), and the complete
[gallery](../results/cage_adaptation/figures/README.md). Component differences,
all-path distributions, validation sensitivity and fixed-final-target
diagnostics support the interpretation without competing for main-text space.

## What each graph measures

| Quantity | Definition and unit | Interpretation |
|---|---|---|
| Native mean cost `J_m` | Sum of four native cost components per simulator step, averaged across rounds/tables/paths | Absolute scalar cost; lower is better |
| Primary difference `Delta J_b` | `J_ours - J_b` | Negative favors ours; original simultaneous two-contrast intervals |
| Cumulative difference `Delta C_b(t)` | `50 * sum_{s<=t}(c_ours(s)-c_b(s))` | Expected native cost over separately reset 50-step episodes; negative favors ours |
| Component difference | Our native component cost minus the same comparator's component | A tradeoff decomposition of the scalar difference |
| Full-target distance `delta_t` | Euclidean distance from average normalized training payoff to full `S(Q_t)` | Absolute vector distance; not a paired native cost difference |
| Fixed-final-target distance | The same average payoff's distance to `S(Q_T)` | Retrospective diagnostic used only for evaluation |
| Validation sensitivity | Mean validation cost for every original `(W,rho)` setting | Explains the earlier parameter choice; no final-test retuning |

Captions must identify the comparator, sign, sampling units, phase boundaries,
uncertainty coverage and normalization. Native losses are not divided by four
for reporting. A cumulative meta-round is an expected mixture of complete
reset episodes; it is not one persistent network deployment.

## Claims supported by the current results

The selected scalar-aware one-switch has 5.3% lower native mean cost than tuned
Hedge and 0.24% higher cost than window in this fixed protocol. All 50 test
paths have identical ours/window actions after round one, so the window
contrast reflects initialization. No safe switching occurred. The additional
geometry checks are conditional on the known normalized training game, with
three modes and affine dimension at most two. At horizons of at least 128,
the selected method and window reach numerical precision on all new paths;
an empirical decay exponent for them is unidentified.

CAGE is the simulator, not a competing learning algorithm. The selected
oracle is an experimental scalar-aware choice within the one-switch master.
These tests do not reproduce the original CAGE competition leaderboard,
evaluate learned native cyber tactics, compare with Marinov et al.'s
implementation, validate the q=4 exponent, or demonstrate an advantage from
the safe fallback. No broader superiority claim should be added.

## Anonymous and public artifacts

For review, upload the anonymous ZIP generated by
`scripts/package_aamas_supplement.py`. It omits identifying public documentation,
Git metadata and archive URLs. Frozen protocols, selection and scientific arrays
are kept byte for byte. Only the primary analysis's non-scientific
`publication_scope` field is removed in an explicitly recorded derivative;
all numerical fields are unchanged. The generic pre-recorded workspace path
in the frozen protocol contains no author identity and is retained to keep its
checksum. The README explains replay from relative package paths.

For the public camera-ready artifact, retain this bilingual repository and
freeze a release/commit; a persistent archive such as Zenodo can give a stable
citation. The official instructions name GitHub and Zenodo as options, without
making a particular hosting service mandatory. The identifying repository
link should be added only to the publication version, not the blind excerpt.
