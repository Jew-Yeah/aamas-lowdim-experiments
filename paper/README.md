[English](README.md) | [Русский](README.ru.md)

# AAMAS 2027 experimental section

This directory contains English LaTeX ready for integration with the official
AAMAS template. The original mathematical manuscript remains outside this
repository and has not been changed.

The recommended main-text selection is [experiments.tex](experiments.tex):
one primary table and one two-panel cumulative paired-cost figure. The figure
shows **our method minus window** and **our method minus Hedge**. Negative
values mean lower cost for our method. It includes setup, observation order,
independent validation/test splits, selected baselines, statistical coverage,
limitations and reproducibility information in the main section.

[supplementary_methods.tex](supplementary_methods.tex) expands the protocol
and includes primary/component differences and absolute full-target geometry.
[experiment_references.bib](experiment_references.bib) supplies the CAGE
citation recommended by the official simulator repository. Merge this entry
into the manuscript's bibliography; keep a single bibliography in the paper.

Copy the English PDFs from [the publication figure pack](../results/cage_adaptation/aamas/README.md)
into the manuscript's `figures/` directory and insert:

```latex
\input{experiments}
```

Use the unchanged official `aamas.cls` and its required packages. Each figure
has a caption beneath it and a plain-text accessibility `\Description`; the
table caption is above the table. No margins, line spacing or document fonts
are changed. The optional geometry figure can be promoted to main text only
if the full manuscript remains within the eight-page content limit.

The retained latest manuscript already fills eight content pages, followed by
references. Adding this section requires allocating space through a separate
whole-paper edit; these fragments alone do not verify the combined page count.
Preserve essential theorem statements and proofs when making that allocation.

For blind submission, refer to the anonymous supplementary ZIP rather than
the identifying public repository. Build it with:

```bash
python scripts/package_aamas_supplement.py --output results/runs/aamas_supplement.zip --ai-statement paper/ai_assistance.md --paper-dir paper
```

The package has a conservative 25,000,000-byte limit and retains frozen
scientific data and code. Read [the submission guidance](../docs/aamas_submission_en.md)
and [AI methodology disclosure](ai_assistance.md). A public repository remains
useful for the archival camera-ready release.
