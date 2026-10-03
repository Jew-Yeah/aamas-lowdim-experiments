"""Anonymous packaging must preserve science while rejecting identifying bytes."""
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
import io
import json
from pathlib import Path
import zipfile
import zlib

import numpy as np
import pytest

SPEC = spec_from_file_location("aamas_supplement", Path(__file__).resolve().parents[1] / "scripts/package_aamas_supplement.py")
module = module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture_repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    names = ["docs/cage_adaptation_protocol.json", "results/cage_adaptation/selection.json",
             "results/cage_adaptation/geometry/primary_sources.json", "data/bank.npz"]
    for name in names:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / names[0]).write_bytes(b'{"article":{"path":"C:/Math/aamas_revision/main.tex"}}\r\n')
    (root / names[1]).write_bytes(b'{"selected":{"scalar":{"window":16,"rho":0.25}}}\n')
    (root / names[2]).write_bytes(b'{"source_checkpoint_sha256":{"seed_1.npz":"abc"}}\r\n')
    np.savez_compressed(root / names[3], values=np.arange(8), labels=np.array(["host_compromise"]))
    analysis = {"primary": {"mean": 1.104, "contrast": -.0618}, "protocol_sha256": "frozen",
                "publication_scope": {"archive_url": "https://github.com/Jew-Yeah/aamas-lowdim-experiments"}}
    analysis_path = root / "results/cage_adaptation/analysis.json"
    analysis_path.write_bytes(module.canonical_json(analysis))
    monkeypatch.setattr(module, "allowlist", lambda: names)
    return root, names, analysis


def test_analysis_projection_removes_only_presentation_fields():
    original = {"primary": {"methods": [1., 2.], "seeds": [4, 5]}, "frozen_at": "original",
                "publication_scope": {"archive_url": "https://github.com/Jew-Yeah/aamas-lowdim-experiments"}}
    raw = module.canonical_json(original)
    projected, provenance = module.anonymous_analysis(raw)
    assert json.loads(projected) == {key: value for key, value in original.items() if key != "publication_scope"}
    assert provenance["original_sha256"] == hashlib.sha256(raw).hexdigest()
    assert provenance["derivative_sha256"] == hashlib.sha256(projected).hexdigest()
    assert provenance["scientific_fields_unchanged"] is True


def test_package_preserves_frozen_crlf_arrays_and_selection_and_is_deterministic(tmp_path, monkeypatch):
    root, names, analysis = fixture_repository(tmp_path, monkeypatch)
    before = {name: (root / name).read_bytes() for name in names}
    result1 = module.build_supplement(root, tmp_path / "one.zip")
    result2 = module.build_supplement(root, tmp_path / "two.zip")
    assert result1 == result2
    assert (tmp_path / "one.zip").read_bytes() == (tmp_path / "two.zip").read_bytes()
    with zipfile.ZipFile(tmp_path / "one.zip") as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        for name, content in before.items():
            assert archive.read(name) == content
            assert (root / name).read_bytes() == content
        assert json.loads(archive.read("results/cage_adaptation/analysis.json"))["primary"] == analysis["primary"]
        for name, value in manifest["files_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == value
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert "WORK_LOG.md" not in archive.namelist()


@pytest.mark.parametrize("text", ["https://github.com/Jew-Yeah/aamas-lowdim-experiments",
                                  "C:/Users/NamedAuthor/project", "/home/named_author/project",
                                  "named_author@example.org", "codex://threads/private-id"])
def test_identifying_text_is_rejected(text):
    with pytest.raises(ValueError, match="Identifying"):
        module.scan_member("README.md", text.encode())


def test_embedded_unicode_array_and_compressed_pdf_metadata_are_scanned():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, metadata=np.array(["C:/Users/NamedAuthor/private"]))
    with pytest.raises(ValueError, match="Identifying"):
        module.scan_member("data.npz", buffer.getvalue())
    pdf = b"%PDF-1.4\nstream\n" + zlib.compress(b"/Author (Jew-Yeah)") + b"\nendstream"
    with pytest.raises(ValueError, match="Identifying"):
        module.scan_member("figure.pdf", pdf)


def test_numeric_binary_is_not_misread_as_personal_text_and_object_arrays_fail():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, numeric=np.frombuffer(b"email@example.org", dtype=np.uint8))
    module.scan_member("scientific.npz", buffer.getvalue())
    objects = io.BytesIO()
    np.savez_compressed(objects, metadata=np.array([{"identity": "named"}], dtype=object))
    with pytest.raises(ValueError, match="Object"):
        module.scan_member("unsafe.npz", objects.getvalue())


@pytest.mark.parametrize("name", ["../outside.txt", "/absolute.txt", "C:/outside.txt", ".git/config", "a\\b"])
def test_archive_paths_cannot_escape_or_include_git_metadata(name):
    with pytest.raises(ValueError, match="Unsafe"):
        module.safe_member(name)


def test_missing_allowlisted_file_fails_and_oversize_preserves_existing_output(tmp_path, monkeypatch):
    root, names, _ = fixture_repository(tmp_path, monkeypatch)
    output = tmp_path / "supplement.zip"
    output.write_bytes(b"previous valid archive")
    with pytest.raises(ValueError, match="exceeds"):
        module.build_supplement(root, output, max_bytes=100)
    assert output.read_bytes() == b"previous valid archive"
    (root / names[0]).unlink()
    with pytest.raises(ValueError, match="Missing"):
        module.build_supplement(root, output)


def test_optional_anonymous_paper_inputs_are_included_and_identifying_readme_is_rejected(tmp_path, monkeypatch):
    root, _, _ = fixture_repository(tmp_path, monkeypatch)
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "experiments.tex").write_text("\\section{Experiments}\n", encoding="utf-8")
    ai = tmp_path / "ai.md"
    ai.write_text("AI assistance was used for code and language editing.\n", encoding="utf-8")
    module.build_supplement(root, tmp_path / "anonymous.zip", paper_dir=paper, ai_statement=ai)
    with zipfile.ZipFile(tmp_path / "anonymous.zip") as archive:
        assert archive.read("paper/experiments.tex") == (paper / "experiments.tex").read_bytes()
        assert archive.read("AI_DISCLOSURE.md") == ai.read_bytes()
    readme = tmp_path / "reviewer.md"
    readme.write_text("Contact author@example.org", encoding="utf-8")
    with pytest.raises(ValueError, match="Identifying"):
        module.build_supplement(root, tmp_path / "unsafe.zip", reviewer_readme=readme)


def test_default_allowlist_contains_manuscript_builder_test_and_english_exports():
    names = module.allowlist()
    assert "scripts/build_aamas_paper_figures.py" in names
    assert "tests/test_aamas_paper_figures.py" in names
    assert "results/cage_adaptation/aamas/provenance.json" in names
    assert "results/cage_adaptation/aamas/captions.tex" in names
    for figure in module.AAMAS_FIGURES:
        for extension in ("pdf", "png"):
            assert f"results/cage_adaptation/aamas/{figure}.{extension}" in names
    assert not any(".ru." in name or name == "WORK_LOG.md" for name in names)


def test_default_allowlist_covers_every_switching_replay_and_its_reproduction_inputs():
    names = set(module.allowlist())
    for name in ("src/lowdim_games/switching_tracking.py", "src/lowdim_games/switching_stress.py",
                 "scripts/run_switching_study.py", "tests/test_switching_tracking.py",
                 "tests/test_switching_stress.py", "docs/switching_en.md",
                 "data/switching_nyc/provenance.json", "data/switching_nyc/nyc_forestry_hazard_daily_2021.csv",
                 "data/switching_nyc/nyc_forestry_hazard_daily_2022.csv", "results/switching/analysis.json",
                 "results/switching/protocol.json", "results/switching/manifest.json"):
        assert name in names
    for study in ("nyc_2021_2022", "nyc_2021", "nyc_2022", "original_budget_stress"):
        assert f"results/switching/{study}/summary.json" in names
        for method in ("one_switch", "shared_past_hull", "lag_safe", "last_window", "block_safe"):
            assert f"results/switching/{study}/{method}.npz" in names
        if study.startswith("nyc_"):
            assert f"results/switching/{study}/game_path.npz" in names
    for figure in ("switching_nyc_dynamics", "switching_original_budget"):
        for extension in ("pdf", "png"):
            assert f"results/switching/figures/{figure}.{extension}" in names
    assert "docs/switching_ru.md" not in names


def test_english_document_projection_preserves_science_and_records_only_navigation_changes():
    raw = ("[English](switching_en.md) | [Русский](switching_ru.md) | [Home](../README.md)\r\n\r\n"
           "# Switching\r\n\r\nThe fixed budget is G=1; delta=0.0009398852984031934.\r\n"
           "[Public code](https://github.com/Jew-Yeah/aamas-lowdim-experiments/tree/main).\r\n").encode()
    projected, details = module.anonymous_document("docs/switching_en.md", raw)
    assert projected == ("# Switching\r\n\r\nThe fixed budget is G=1; delta=0.0009398852984031934.\r\n"
                         "[Public code](../README.md).\r\n").encode()
    assert details["original_sha256"] == module.digest(raw)
    assert details["derivative_sha256"] == module.digest(projected)
    assert details["removed_language_navigation"] is True
    assert details["author_repository_links_replaced"] == 1
    assert details["scientific_prose_and_values_unchanged"] is True
    module.scan_member("docs/switching_en.md", projected)
    ordinary = b"# Fixed trace\r\nRaw signed errors are retained.\r\n"
    assert module.anonymous_document("results/switching/README.md", ordinary)[0] == ordinary


def test_excluded_presentation_links_are_redirected_to_included_sources_without_changing_text():
    raw = ("G=1; all five methods are retained.\n"
           "[AAMAS integration](../../paper/README.md), "
           "[frozen CAGE comparison](../cage_adaptation/README.md).\n").encode()
    projected, details = module.anonymous_document("results/switching/README.md", raw)
    assert projected == ("G=1; all five methods are retained.\n"
                         "[AAMAS integration](../../paper/experiments.tex), "
                         "[frozen CAGE comparison](../cage_adaptation/analysis.json).\n").encode()
    assert details["author_repository_links_replaced"] == 0
    assert details["removed_language_navigation"] is False
    assert sum(row["replacements"] for row in details["excluded_document_links_redirected"]) == 2


def add_switching_fixture(root, names, monkeypatch):
    """Small complete freeze with real checksum links, without replaying learners."""
    source_names = ["scripts/run_switching_study.py", "src/lowdim_games/switching_tracking.py",
                    "src/lowdim_games/switching_stress.py", "src/lowdim_games/learners.py",
                    "src/lowdim_games/geometry.py", "tests/test_switching_tracking.py",
                    "tests/test_switching_stress.py"]
    added = module.switching_allowlist() + source_names
    for name in added:
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".npz":
            np.savez_compressed(source, payoff=np.array([-.1, .2]), mode=np.array(["fast", "safe"]))
        elif source.suffix == ".pdf":
            source.write_bytes(b"%PDF-1.4\n% Anonymous English figure\n")
        elif source.suffix == ".png":
            source.write_bytes(b"\x89PNG\r\n\x1a\n")
        elif source.suffix == ".json":
            source.write_bytes(b'{}\n')
        elif source.suffix == ".csv":
            source.write_bytes(b"date,Bronx,Brooklyn,Manhattan,Queens,Staten Island\r\n2021-01-01,1,2,3,4,5\r\n")
        elif source.suffix == ".md":
            source.write_text("[English](switching_en.md) | [Русский](switching_ru.md)\n\n"
                              "# Fixed switching studies\nAll five methods are retained.\n", encoding="utf-8")
        else:
            source.write_bytes(b"# Frozen causal learner/source check\r\n")
    provenance = {"years": {}}
    for year in ("2021", "2022"):
        filename = f"nyc_forestry_hazard_daily_{year}.csv"
        provenance["years"][year] = {"csv": filename,
                                     "csv_sha256": module.digest((root / "data/switching_nyc" / filename).read_bytes())}
    (root / "data/switching_nyc/provenance.json").write_bytes(module.canonical_json(provenance))
    protocol = {"source_sha256": {name: module.digest((root / name).read_bytes()) for name in source_names[:5]}}
    (root / "results/switching/protocol.json").write_bytes(module.canonical_json(protocol))
    # Include this optional report in the result freeze too, to check that hashes
    # are verified against source bytes before its navigation derivative is made.
    report = "results/switching/README.md"
    (root / report).write_text("# Results\n[Public code](https://github.com/Jew-Yeah/aamas-lowdim-experiments).\n",
                              encoding="utf-8")
    result_names = [name for name in added + [report] if name.startswith("results/switching/")
                    and name != "results/switching/manifest.json"]
    manifest = {"files_sha256": {name.removeprefix("results/switching/"): module.digest((root / name).read_bytes())
                                for name in result_names}}
    (root / "results/switching/manifest.json").write_bytes(module.canonical_json(manifest))
    monkeypatch.setattr(module, "allowlist", lambda: names + added)
    return added


def test_complete_switching_freeze_is_preserved_and_new_paper_figures_match(tmp_path, monkeypatch):
    root, names, _ = fixture_repository(tmp_path, monkeypatch)
    added = add_switching_fixture(root, names, monkeypatch)
    before = {name: (root / name).read_bytes() for name in added}
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "experiments.tex").write_text("\\section{Actual switching}\n", encoding="utf-8")
    (paper / "ai_assistance.md").write_text("AI helped design and implement these exploratory studies.\n", encoding="utf-8")
    output = tmp_path / "switching.zip"
    module.build_supplement(root, output, paper_dir=paper)
    with zipfile.ZipFile(output) as archive:
        for name, original in before.items():
            assert (root / name).read_bytes() == original
            if name not in module.ENGLISH_DOCUMENTS:
                assert archive.read(name) == original
        for figure in module.SWITCHING_FIGURES:
            for extension in ("png", "pdf"):
                assert archive.read(f"paper/figures/{figure}.{extension}") == archive.read(
                    f"results/switching/figures/{figure}.{extension}")
        assert archive.read("AI_DISCLOSURE.md") == archive.read("paper/ai_assistance.md")
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["switching_input_verification"]["aggregate_csv_files_verified"] == 2
        assert manifest["switching_input_verification"]["source_code_files_verified"] == 5
        assert manifest["english_document_projections"]["results/switching/README.md"]["author_repository_links_replaced"] == 1
        for name, expected in manifest["files_sha256"].items():
            assert module.digest(archive.read(name)) == expected
            module.scan_member(name, archive.read(name))
    (paper / "figures").mkdir()
    (paper / "figures/switching_nyc_dynamics.pdf").write_bytes(b"%PDF-1.4\n% altered figure\n")
    with pytest.raises(ValueError, match="differs from the canonical"):
        module.build_supplement(root, output, paper_dir=paper)


@pytest.mark.parametrize("name,role", [
    ("data/switching_nyc/nyc_forestry_hazard_daily_2021.csv", "aggregate CSV"),
    ("src/lowdim_games/switching_tracking.py", "source code"),
    ("results/switching/nyc_2021_2022/one_switch.npz", "result"),
])
def test_switching_checksum_mismatch_fails_without_replacing_existing_archive(tmp_path, monkeypatch, name, role):
    root, names, _ = fixture_repository(tmp_path, monkeypatch)
    add_switching_fixture(root, names, monkeypatch)
    (root / name).write_bytes((root / name).read_bytes() + b"modified")
    output = tmp_path / "existing.zip"
    output.write_bytes(b"previous valid archive")
    with pytest.raises(ValueError, match=f"Switching {role} checksum mismatch"):
        module.build_supplement(root, output)
    assert output.read_bytes() == b"previous valid archive"


def test_conflicting_ai_disclosures_fail_instead_of_shipping_two_versions(tmp_path, monkeypatch):
    root, _, _ = fixture_repository(tmp_path, monkeypatch)
    paper = tmp_path / "paper"
    paper.mkdir()
    (paper / "ai_assistance.md").write_text("Current experimental design disclosure.\n", encoding="utf-8")
    override = tmp_path / "old_ai.md"
    override.write_text("Earlier language-only disclosure.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="AI disclosure differs"):
        module.build_supplement(root, tmp_path / "unsafe.zip", paper_dir=paper, ai_statement=override)


def test_manuscript_exports_are_mirrored_exactly_and_conflicting_figures_fail(tmp_path, monkeypatch):
    root, names, _ = fixture_repository(tmp_path, monkeypatch)
    new_names = []
    for figure in module.AAMAS_FIGURES:
        name = f"results/cage_adaptation/aamas/{figure}.pdf"
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"%PDF-1.4\n% Anonymous vector export\n")
        new_names.append(name)
    monkeypatch.setattr(module, "allowlist", lambda: names + new_names)
    output = tmp_path / "supplement.zip"
    module.build_supplement(root, output)
    with zipfile.ZipFile(output) as archive:
        for name in new_names:
            assert archive.read(name) == archive.read("paper/figures/" + Path(name).name)
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["presentation_provenance_analysis_mapping"]["anonymous_analysis_sha256"] == manifest["analysis_projection"]["derivative_sha256"]
    paper = tmp_path / "paper"
    (paper / "figures").mkdir(parents=True)
    (paper / "figures/primary_comparison.pdf").write_bytes(b"%PDF-1.4\n% altered result\n")
    with pytest.raises(ValueError, match="differs from the canonical"):
        module.build_supplement(root, output, paper_dir=paper)
