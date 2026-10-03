"""Build a deterministic anonymous reviewer ZIP without changing research files.

All selected source, calibration, trajectory, selection and protocol bytes are
copied exactly. Documented derivatives omit the old analysis presentation block
and identifying/navigation links in English documentation. The author repository
and unrelated public presentation are excluded.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile
import zlib

REPO = Path(__file__).resolve().parents[1]
MAX_BYTES = 25_000_000
TEXT_EXTENSIONS = {".py", ".json", ".md", ".txt", ".toml", ".tex", ".bib", ".csv", ".yaml"}
SOURCE_NAMES = (
    "__init__", "benchmarks", "cage", "cage_adaptations", "cage_study", "cli",
    "data", "experiment", "game", "geometry", "learners", "plotting",
    "policy_experiment", "policy_plotting", "recommended", "switching_tracking",
    "switching_stress",
)
SCRIPT_NAMES = ("calibrate_cage", "collect_cage_bank", "run_cage_adaptation_study",
                "build_cage_dynamic_figures", "build_cage_geometry_figures", "build_aamas_paper_figures",
                "run_switching_study")
TEST_NAMES = ("benchmarks", "cage", "cage_adaptation_study", "cage_adaptations",
              "cage_bank", "cage_dynamic_figures", "cage_geometry_figures", "cage_study",
              "data", "game", "geometry", "learners", "policy_experiment", "protocol", "recommended",
              "aamas_paper_figures", "switching_tracking", "switching_stress")
REPORT_FILES = ("protocol.json", "selection.json", "selection.sha256.json",
                "validation_grid.json", "trace_diagnostics.json")
GEOMETRY_FILES = ("protocol.json", "serialization_repair.json", "primary_sources.json",
                  "primary_inputs.npz", "prefix_geometry.npz", "horizon_geometry.npz",
                  "horizon_inputs.npz", "horizon_sources.json", "metrics.json")
DYNAMIC_FILES = ("protocol.json", "summary.json", "plot_data.npz")
GEOMETRY_FIGURES = ("prefix_error", "prefix_fixed_target", "horizon_error")
DYNAMIC_FIGURES = ("loss_dynamics", "cumulative_differences", "loss_components",
                   "paired_path_distribution", "validation_sensitivity")
AAMAS_FIGURES = ("primary_comparison", "calibrated_geometry", "cumulative_cost_difference")
AAMAS_FILES = ("provenance.json", "captions.tex")
SWITCHING_STUDIES = ("nyc_2021_2022", "nyc_2021", "nyc_2022", "original_budget_stress")
SWITCHING_METHODS = ("one_switch", "shared_past_hull", "lag_safe", "last_window", "block_safe")
SWITCHING_FIGURES = ("switching_nyc_dynamics", "switching_original_budget")
ENGLISH_DOCUMENTS = ("docs/switching_en.md", "results/switching/README.md")
ANONYMOUS_DOCUMENT_LINKS = {
    "../../paper/README.md": "../../paper/experiments.tex",
    "../cage_adaptation/README.md": "../cage_adaptation/analysis.json",
}
AUTHOR_REPOSITORY_URL = re.compile(
    r"https?://github\.com/Jew-Yeah/aamas-lowdim-experiments(?:/[^\s)>\]]*)?", re.I
)
FORBIDDEN = (
    re.compile(r"Jew-Yeah|aamas-lowdim-experiments", re.I),
    re.compile(r"(?<![A-Za-z])(?:[A-Za-z]:[/\\](?:Users|Documents and Settings)[/\\])", re.I),
    re.compile(r"/(?:Users|home)/[^/\s]+", re.I),
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?:\.codex[/\\]|codex://|plugin://|threadId|clientThreadId)", re.I),
)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def canonical_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def anonymous_analysis(content):
    """Remove only non-scientific presentation metadata; retain all other values."""
    original = json.loads(content)
    anonymous = deepcopy(original)
    anonymous.pop("publication_scope", None)
    expected = {key: value for key, value in original.items() if key != "publication_scope"}
    if anonymous != expected:
        raise ValueError("Analysis projection changed scientific values.")
    return canonical_json(anonymous), {
        "removed_keys": ["publication_scope"] if "publication_scope" in original else [],
        "original_sha256": digest(content),
        "derivative_sha256": digest(canonical_json(anonymous)),
        "scientific_fields_unchanged": True,
    }


def anonymous_document(name, content):
    """Remove only public navigation; retain all scientific prose and values."""
    text = content.decode("utf-8")
    lines = text.splitlines(keepends=True)
    navigation_removed = bool(lines and lines[0].startswith("[English](")
                              and "[Русский](" in lines[0])
    if navigation_removed:
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
        text = "".join(lines)
    anonymous_readme = "../" * (len(PurePosixPath(name).parts) - 1) + "README.md"
    text, replacement_count = AUTHOR_REPOSITORY_URL.subn(anonymous_readme, text)
    redirected_links = []
    for original, included in ANONYMOUS_DOCUMENT_LINKS.items():
        target = "](" + original + ")"
        count = text.count(target)
        if count:
            text = text.replace(target, "](" + included + ")")
            redirected_links.append({"original_destination": original, "anonymous_destination": included,
                                     "replacements": count})
    projected = text.encode("utf-8")
    return projected, {
        "original_sha256": digest(content),
        "derivative_sha256": digest(projected),
        "removed_language_navigation": navigation_removed,
        "author_repository_links_replaced": replacement_count,
        "replacement_destination": anonymous_readme if replacement_count else None,
        "excluded_document_links_redirected": redirected_links,
        "scientific_prose_and_values_unchanged": True,
    }


def safe_member(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts
            or any(part in {".git", ".external", ".venv", ".venv-cage", "__pycache__"} for part in path.parts)):
        raise ValueError(f"Unsafe archive member: {name}")
    return name


def scan_text(name, text):
    for pattern in FORBIDDEN:
        if pattern.search(text):
            raise ValueError(f"Identifying content in {name}: {pattern.pattern}")


def scan_member(name, content):
    """Check text, embedded array strings and PDF text/metadata streams."""
    safe_member(name)
    scan_text(name, name)
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        scan_text(name, content.decode("utf-8-sig"))
    elif suffix == ".npz":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for member in archive.infolist():
                payload = archive.read(member)
                if not payload.startswith(b"\x93NUMPY"):
                    raise ValueError(f"Invalid NPY member: {name}:{member.filename}")
                width = 2 if payload[6] == 1 else 4
                start = 8 + width
                size = int.from_bytes(payload[8:start], "little")
                header = ast.literal_eval(payload[start:start + size].decode("latin-1"))
                descriptor = header["descr"]
                if not isinstance(descriptor, str) or "O" in descriptor:
                    raise ValueError(f"Object or structured NPY data are not allowed: {name}:{member.filename}")
                if not re.match(r"^[<>=|][US]", descriptor):
                    # Numeric array bytes can accidentally resemble email addresses;
                    # they contain no textual identity fields and are not scanned.
                    continue
                # Numpy Unicode arrays use UTF-32: removing NUL bytes also
                # exposes ASCII identifiers embedded in those string arrays.
                payload = payload[start + size:].replace(b"\0", b"")
                scan_text(name + ":" + member.filename, payload.decode("utf-8", errors="ignore"))
    elif suffix == ".pdf":
        scan_text(name, content.decode("latin-1"))
        for value in re.findall(rb"<([0-9a-fA-F]{8,})>", content):
            try:
                raw = bytes.fromhex(value.decode("ascii"))
            except ValueError:
                continue
            scan_text(name, raw.replace(b"\0", b"").decode("utf-8", errors="ignore"))
        for value in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", content, flags=re.S):
            try:
                decoded = zlib.decompress(value)
            except zlib.error:
                continue
            scan_text(name, decoded.replace(b"\0", b"").decode("utf-8", errors="ignore"))
    elif suffix == ".png":
        # Scan textual metadata, not compressed pixel bytes.
        position = 8
        while position + 12 <= len(content):
            size = int.from_bytes(content[position:position + 4], "big")
            kind = content[position + 4:position + 8]
            payload = content[position + 8:position + 8 + size]
            if kind in {b"tEXt", b"iTXt"}:
                scan_text(name, payload.decode("utf-8", errors="ignore"))
            if kind == b"zTXt" and b"\0" in payload:
                try:
                    decoded = zlib.decompress(payload.split(b"\0", 1)[1][1:])
                    scan_text(name, decoded.decode("utf-8", errors="ignore"))
                except zlib.error:
                    pass
            position += size + 12


def switching_allowlist():
    """Explicit complete inputs/outputs for the four frozen switching replays."""
    names = ["docs/switching_en.md"]
    names += [f"data/switching_nyc/{name}" for name in
              ("provenance.json", "nyc_forestry_hazard_daily_2021.csv", "nyc_forestry_hazard_daily_2022.csv")]
    names += [f"results/switching/{name}" for name in ("analysis.json", "protocol.json", "manifest.json")]
    names += [f"results/switching/{study}/{name}" for study in SWITCHING_STUDIES
              for name in ("summary.json", *(f"{method}.npz" for method in SWITCHING_METHODS))]
    names += [f"results/switching/{study}/game_path.npz" for study in SWITCHING_STUDIES
              if study.startswith("nyc_")]
    names += [f"results/switching/figures/{name}.{extension}"
              for name in SWITCHING_FIGURES for extension in ("png", "pdf")]
    return names


def allowlist():
    names = ["pyproject.toml", "LICENSE", "requirements-lock.txt", "requirements-cage-lock.txt",
             "docs/cage_adaptation_protocol.json", "docs/cage_figures_protocol.json"]
    names += [f"src/lowdim_games/{name}.py" for name in SOURCE_NAMES]
    names += [f"scripts/{name}.py" for name in SCRIPT_NAMES]
    names += [f"tests/test_{name}.py" for name in TEST_NAMES]
    names += [f"data/cage2/{name}" for name in ("calibration.npz", "provenance.json", "cell_statistics.json")]
    names += [f"data/cage2_adaptation/{role}50/{name}" for role in ("validation", "test")
              for name in ("bank.npz", "manifest.json", "cell_statistics.json")]
    names += [f"results/cage_adaptation/{name}" for name in REPORT_FILES]
    names += [f"results/cage_adaptation/groups/{group}/{name}" for group in ("validation", "test")
              for name in ("meta.json", "training_fit.npz", "occupancies.npz")]
    names += [f"results/cage_adaptation/geometry/{name}" for name in GEOMETRY_FILES]
    names += [f"results/cage_adaptation/dynamics/{name}" for name in DYNAMIC_FILES]
    names += [f"results/cage_adaptation/{name}.{extension}" for name in ("primary_means", "primary_comparisons")
              for extension in ("png", "pdf")]
    names += [f"results/cage_adaptation/{directory}/{name}.{extension}"
              for directory, figures in (("geometry", GEOMETRY_FIGURES), ("dynamics", DYNAMIC_FIGURES))
              for name in figures for extension in ("png", "pdf")]
    names += [f"results/cage_adaptation/aamas/{name}" for name in AAMAS_FILES]
    names += [f"results/cage_adaptation/aamas/{name}.{extension}"
              for name in AAMAS_FIGURES for extension in ("png", "pdf")]
    return sorted(names + switching_allowlist())


def read_source(repo, name):
    source = repo / safe_member(name)
    if source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(repo.resolve()):
        raise ValueError(f"Missing or unsafe allowlisted source: {name}")
    return source.read_bytes()


def verify_switching_inputs(members):
    """Fail before writing if cached results, aggregate inputs or code changed."""
    manifest_name = "results/switching/manifest.json"
    if manifest_name not in members:
        return None

    def verify(name, expected, role):
        safe_member(name)
        if name not in members:
            raise ValueError(f"Missing switching {role} input: {name}")
        if digest(members[name]) != expected:
            raise ValueError(f"Switching {role} checksum mismatch: {name}")

    recorded = json.loads(members[manifest_name])["files_sha256"]
    for name, expected in recorded.items():
        verify("results/switching/" + safe_member(name), expected, "result")
    provenance = json.loads(members["data/switching_nyc/provenance.json"])
    for year in ("2021", "2022"):
        row = provenance["years"][year]
        verify("data/switching_nyc/" + safe_member(row["csv"]), row["csv_sha256"], "aggregate CSV")
    protocol = json.loads(members["results/switching/protocol.json"])
    for name, expected in protocol["source_sha256"].items():
        verify(name, expected, "source code")
    return {"result_manifest": manifest_name, "result_files_verified": len(recorded),
            "aggregate_csv_files_verified": 2, "source_code_files_verified": len(protocol["source_sha256"])}


DEFAULT_README = """# Anonymous reproducibility supplement

The main experimental section now examines an actual one-switch crossing on
730 chronological days of NYC Hazard request shares, using a separately certified
lag safe base. Two fresh annual checks and a constructed original-block-budget
diagnostic are included. Each replay reports all five causal methods and the full
fast/safe trajectory. These are exploratory fixed traces, without confidence
intervals or a claim of universal superiority. See docs/switching_en.md, frozen
results/switching/analysis.json, and the included experimental LaTeX.

The earlier CAGE policy-selection study is retained as secondary evidence with
its fixed calibrated game, validation and held-out simulator seed tables, locked
selection, all 50 test paths and separate 120-path horizon sweep. It does not
activate the safe switch. No new tuning, data download or simulator collection
is needed to inspect or reproduce the cached studies and English figures.

Use Python 3.12. The recorded implementation was tested with the pinned versions
in requirements-lock.txt. From the extracted archive directory:

```sh
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
python scripts/run_switching_study.py --output results/runs/switching_rebuild
python -m lowdim_games.cli cage-selected --horizon 16 --seeds 41000000 --output results/runs/smoke
python scripts/build_cage_dynamic_figures.py --output results/runs/dynamic_rebuild
python scripts/build_cage_geometry_figures.py --stage report
python scripts/build_aamas_paper_figures.py --output results/runs/aamas_presentation_rebuild
```

The switching command verifies both cached aggregate CSV hashes, reruns the
combined NYC trace, both annual traces and original-budget diagnostic, and exports
English PNG/PDF dynamics into a separate directory. It requires no network access.
Raw arrays retain values below the disclosed NYC plot display floor of 1e-5.
Average-payoff distance and mean daily payoff norm/absolute imbalance are distinct
metrics; cumulative difference panels identify their comparator and sign.

The short CAGE command is a functional check, not the final statistical experiment.
Omit --horizon and --seeds to replay all three selected methods on all 50 final
paths. The geometry prefix stage can recompute every original-path projection;
the all stage also regenerates the independently announced horizon sweep.

Calibration and all episodes come from the official CAGE Challenge 2 simulator,
revision 26ce1c1253fa9e2e73f25e6a7f2da32860c11257. Its source URL is preserved in
data provenance. The simulator itself is excluded; it is unnecessary for the
cached empirical-model replay. Recollecting simulator episodes is a separate task
requiring the upstream checkout and requirements-cage-lock.txt.

The full original protocols, trajectories and locked selection retain exact source
bytes. Before packaging, the switching result manifest, code hashes and both CSV
hashes are checked. The NYC provenance retains official dataset/query URLs and
the exact aggregate hashes; the counts are registered requests, not measured
staffing demand or realized service quality. The constructed stress uses
payoff-equivalent orthogonal labels and is a mechanism diagnostic, not a realistic
attacker-learning or q=4 rate experiment.

The protocol records a generic workspace path for the unchanged manuscript;
that historical path contains no identity and is not required for reproduction.
Analysis.json is an explicitly documented derivative: only publication_scope
(public presentation/archive metadata) is omitted. All scientific fields are
unchanged. MANIFEST.json records its original and derivative SHA-256 hashes and
the exact archive checksum of every other member. No original-analysis checksum
verification is claimed for the derivative. The original mixed policy actions,
seeds, numerical results, calibration, selection and protocols are unchanged.
English documentation derivatives remove only the language-navigation row and
replace any identifying author-repository links with the local anonymous README.
Links to excluded presentation READMEs point to the included experiment source
and frozen CAGE analysis instead.
MANIFEST.json records both hashes and the precise presentation-only operations.
The exact original AAMAS presentation provenance is also retained. Its historical
analysis input checksum refers to the original analysis, linked to the anonymous
derivative in MANIFEST.json. Rebuilding presentation uses the derivative checksum:
the figures remain byte-identical under the recorded software, while regenerated
presentation provenance records this different non-scientific input identity.

In the secondary CAGE study, loss intervals are conditional on calibration and
locked parameters. Extra curve
intervals are descriptive and pointwise. Vector error uses the calibrated game
and the full realized-hull response target, not held-out native loss. Numerical
checks are floating-point evidence. Near-zero errors cannot identify a decay
exponent; a display-scaled T^(-1/2) guide is not a fitted theoretical constant.

The public presentation, identifying repository links, Git metadata, local tools,
environments and unrelated results are excluded. Optional manuscript fragments
and an AI disclosure, when supplied, are placed in paper/ and AI_DISCLOSURE.md.
"""


def deterministic_zip(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(safe_member(name), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def build_supplement(repo, output, *, reviewer_readme=None, ai_statement=None,
                     paper_dir=None, max_bytes=MAX_BYTES):
    repo, output = Path(repo), Path(output)
    members, sources = {}, {}
    for name in allowlist():
        content = read_source(repo, name)
        members[name] = content
        sources[name] = {"sha256": digest(content), "source_bytes_unchanged": True}
    optional_report = "results/switching/README.md"
    if (repo / optional_report).is_file():
        members[optional_report] = read_source(repo, optional_report)
        sources[optional_report] = {"sha256": digest(members[optional_report]), "source_bytes_unchanged": True}
    switching_verification = verify_switching_inputs(members)
    document_projections = {}
    for name in ENGLISH_DOCUMENTS:
        if name in members:
            projected, details = anonymous_document(name, members[name])
            members[name] = projected
            sources[name] = {"sha256": digest(projected),
                             "source_bytes_unchanged": details["original_sha256"] == details["derivative_sha256"]}
            document_projections[name] = details
    analysis_name = "results/cage_adaptation/analysis.json"
    projected, projection = anonymous_analysis(read_source(repo, analysis_name))
    members[analysis_name] = projected
    sources[analysis_name] = {"sha256": digest(projected), "source_bytes_unchanged": False}
    members["README.md"] = (Path(reviewer_readme).read_bytes() if reviewer_readme
                            else DEFAULT_README.encode("utf-8"))
    if ai_statement:
        members["AI_DISCLOSURE.md"] = Path(ai_statement).read_bytes()
    if paper_dir:
        paper_dir = Path(paper_dir)
        for name in ("experiments.tex", "supplement.tex", "supplementary_methods.tex",
                     "experiment_references.bib", "ai_assistance.md"):
            source = paper_dir / name
            if source.is_file():
                if source.is_symlink():
                    raise ValueError("Do not package symbolic links.")
                members["paper/" + name] = source.read_bytes()
        if "paper/ai_assistance.md" in members:
            if "AI_DISCLOSURE.md" in members and members["AI_DISCLOSURE.md"] != members["paper/ai_assistance.md"]:
                raise ValueError("Supplied AI disclosure differs from the current paper disclosure.")
            members["AI_DISCLOSURE.md"] = members["paper/ai_assistance.md"]
        figures = paper_dir / "figures"
        if figures.is_dir():
            for source in sorted(figures.iterdir()):
                if source.is_file() and source.suffix.lower() in {".png", ".pdf"}:
                    if source.is_symlink():
                        raise ValueError("Do not package symbolic links.")
                    members["paper/figures/" + source.name] = source.read_bytes()
    # Mirror the exact English manuscript exports beside the insertable sources.
    # Their canonical source location and original provenance remain included.
    for directory, names in (("results/cage_adaptation/aamas", AAMAS_FIGURES),
                             ("results/switching/figures", SWITCHING_FIGURES)):
        for name in names:
            for extension in ("png", "pdf"):
                source_name = f"{directory}/{name}.{extension}"
                if source_name in members:
                    destination = f"paper/figures/{name}.{extension}"
                    if destination in members and members[destination] != members[source_name]:
                        raise ValueError(f"Supplied paper figure differs from the canonical export: {destination}")
                    members[destination] = members[source_name]
    for name, content in members.items():
        scan_member(name, content)
    manifest = {
        "schema_version": 1,
        "anonymous_reviewer_package": True,
        "deterministic_zip_timestamp": "1980-01-01T00:00:00",
        "maximum_bytes": max_bytes,
        "analysis_projection": projection,
        "english_document_projections": document_projections,
        "switching_input_verification": switching_verification,
        "experimental_scope": {
            "main": "Exploratory chronological NYC request-share tracking with a certified lag safe base",
            "additional": "Constructed original-block-budget mechanism diagnostic and overlapping annual NYC replays",
            "secondary_frozen": "Original CAGE policy-selection study; no safe switching",
        },
        "presentation_provenance_analysis_mapping": {
            "provenance_file": "results/cage_adaptation/aamas/provenance.json",
            "recorded_original_analysis_sha256": projection["original_sha256"],
            "anonymous_analysis_sha256": projection["derivative_sha256"],
            "original_provenance_bytes_retained": True,
            "rebuild_effect": "Figures unchanged; regenerated presentation provenance references the anonymous derivative checksum",
        },
        "frozen_scientific_source_files": sources,
        "files_sha256": {name: digest(content) for name, content in sorted(members.items())},
        "excluded": ["author repository links", "public presentation documents", "Git metadata",
                     "local environments", "unrelated results", "Russian duplicate figures"],
    }
    members["MANIFEST.json"] = canonical_json(manifest)
    scan_member("MANIFEST.json", members["MANIFEST.json"])
    payload = deterministic_zip(members)
    if len(payload) > max_bytes:
        raise ValueError(f"Supplement ZIP exceeds {max_bytes} bytes: {len(payload)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".supplement-", suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"files": len(members), "bytes": len(payload), "sha256": digest(payload),
            "analysis_projection": projection}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output", type=Path, default=REPO / "results/runs/aamas_supplement.zip")
    parser.add_argument("--reviewer-readme", type=Path)
    parser.add_argument("--ai-statement", type=Path)
    parser.add_argument("--paper-dir", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_supplement(args.repo, args.output, reviewer_readme=args.reviewer_readme,
                                     ai_statement=args.ai_statement, paper_dir=args.paper_dir), indent=2))


if __name__ == "__main__":
    main()
